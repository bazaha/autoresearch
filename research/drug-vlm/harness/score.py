"""
Fixed scoring harness for the drug-VLM autoresearch loop (reference implementation).
This is the equivalent of `evaluate_bpb` in prepare.py: deterministic, CPU-only,
and READ-ONLY for the agent. The agent edits train.py; this file decides keep/discard.

Gold and predictions are JSONL files, one document (a page, a crop, or a patent) per line:

  {"doc_id": "US1234567B2-p12", "family_id": "US1234567", "slice": "patent",
   "compounds":    [{"id": "12a", "smiles": "CC(=O)Nc1ccc(O)cc1"}],
   "measurements": [{"compound_id": "12a", "target": "EGFR", "endpoint": "IC50",
                     "relation": "=", "value": 12.3, "unit": "nM"},
                    {"compound_id": "12a", "target": "EGFR", "endpoint": "% inhibition",
                     "value": 87, "unit": "%", "conc": 1, "conc_unit": "uM"},
                    {"compound_id": "12a", "target": "hERG", "endpoint": "IC50",
                     "value": [10, 30], "unit": "uM"}]}

`family_id` (patent family / article) is the bootstrap cluster and defaults to doc_id;
`slice` (gold only) groups documents for per-slice reporting. An OCSR crop is simply a
document with one compound and no measurements. A value given as [lo, hi] is a range.

Structures are SMILES/CXSMILES. Markush/R-group positions are dummy atoms written as
`[1*]`, `[*:1]` or a CXSMILES label (`*... |$R1$|`, `_R1`); all three mean R1. Racemic or
relative stereo uses CXSMILES enhanced stereo (`|&1:1|`, `|o1:1|`).

Usage:
    python score.py score   --gold gold.jsonl --pred pred.jsonl [--per-doc-out per_doc.jsonl]
    python score.py compare --base seed0.jsonl,seed1.jsonl --new seed0.jsonl,seed1.jsonl
    (compare takes per-doc files written by `score`; several seeds per side are pooled)
"""

import argparse
import json
import math
import os
import re
import sys
from functools import lru_cache

import numpy as np
from rdkit import Chem, DataStructs, RDLogger, __version__ as RDKIT_VERSION
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog("rdApp.*")

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify — changing these changes the metric)
# ---------------------------------------------------------------------------

RDKIT_PIN = "2026.09.1"  # canonicalization changes between RDKit releases; scores are only comparable on one version
W_STRUCT = 0.5          # weight of structure F1 in the composite score
W_RECORD = 0.5          # weight of record F1 (structure + value correctly linked)
VALUE_REL_TOL = 1e-3    # relative tolerance when gold and prediction use the same scale
LOG_TOL_CONVERTED = 0.05  # log10 tolerance when one side converted pX <-> molar (rounding)
STRIP_SALTS = True      # compare parent compounds (largest fragment, neutralized)
MATCH_TARGET = True     # a record must also name the right target
MAX_HEAVY_ATOMS = 500   # larger predicted structures are invalid (keeps scoring time bounded)
MAX_PRED_COMPOUNDS = 5000   # per document; the rest count as invalid predictions
MAX_PRED_RECORDS = 20000    # per document; the rest count as false positives
BOOTSTRAP_SAMPLES = 1000
BOOTSTRAP_SEED = 0

# ---------------------------------------------------------------------------
# Structure normalization
# ---------------------------------------------------------------------------

_morgan = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
_uncharger = rdMolStandardize.Uncharger()
_salt_remover = rdMolStandardize.FragmentRemover()
_GENERIC_LABELS = {"X": 90, "Y": 91, "Z": 92, "A": 93, "Ar": 94, "Het": 95, "Alk": 96, "Q": 97, "G": 98}
_DUMMY_PROPS = ("atomLabel", "dummyLabel", "_MolFileRLabel")


def _rlabel_number(atom):
    """R-group number of a dummy atom from [1*], [*:1], CXSMILES/molfile labels; 0 for a bare '*'."""
    if atom.GetIsotope():
        return atom.GetIsotope()
    if atom.GetAtomMapNum():
        return atom.GetAtomMapNum()
    props = atom.GetPropsAsDict(includePrivate=True, includeComputed=False)
    if props.get("_MolFileRLabel"):
        return int(props["_MolFileRLabel"])
    label = str(props.get("atomLabel") or props.get("dummyLabel") or "*")
    m = re.fullmatch(r"_?R(\d+)", label)
    if m:
        return int(m.group(1))
    return _GENERIC_LABELS.get(label.lstrip("_"), 0 if label in ("*", "R", "_R") else 99)


@lru_cache(maxsize=200_000)
def _standardize(smiles):
    """SMILES/CXSMILES -> standardized RDKit mol, or None if unparseable."""
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    mol = Chem.MolFromSmiles(smiles.strip())
    if mol is None:
        return None
    try:
        mol = Chem.RWMol(mol)
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 0:   # one spelling for every R-label convention
                n = _rlabel_number(atom)
                for prop in _DUMMY_PROPS:
                    if atom.HasProp(prop):
                        atom.ClearProp(prop)
                atom.SetIsotope(n)
            atom.SetAtomMapNum(0)
        mol = mol.GetMol()
        if STRIP_SALTS:
            mol = rdMolStandardize.FragmentParent(mol)
            mol = _uncharger.uncharge(mol)
        else:
            mol = rdMolStandardize.Cleanup(mol)
        # round-trip through canonical CXSMILES: fresh ring info, canonical enhanced-stereo groups
        mol = Chem.MolFromSmiles(Chem.MolToCXSmiles(mol))
    except Exception:
        return None
    if mol is None or mol.GetNumHeavyAtoms() > MAX_HEAVY_ATOMS:
        return None
    return mol


@lru_cache(maxsize=200_000)
def structure_keys(smiles):
    """
    -> (key, skeleton) used for exact matching, or (None, None) for invalid input.
    key: standard InChIKey (stereo-aware) for ordinary molecules; canonical CXSMILES for
    Markush/R-group structures (InChI cannot represent them) and for racemic/relative
    (enhanced) stereo, which standard InChI does not distinguish from absolute stereo.
    skeleton: the same identity with stereochemistry removed (diagnostics only).
    """
    mol = _standardize(smiles)
    if mol is None:
        return None, None
    if any(atom.GetAtomicNum() == 0 for atom in mol.GetAtoms()):
        flat = Chem.Mol(mol)
        Chem.RemoveStereochemistry(flat)
        return "MARKUSH:" + Chem.MolToCXSmiles(mol), "MARKUSH:" + Chem.MolToSmiles(flat)
    inchikey = Chem.MolToInchiKey(mol)
    if not inchikey:
        flat = Chem.Mol(mol)
        Chem.RemoveStereochemistry(flat)
        return "SMILES:" + Chem.MolToCXSmiles(mol), "SMILES:" + Chem.MolToSmiles(flat)
    if any(g.GetGroupType() != Chem.StereoGroupType.STEREO_ABSOLUTE for g in mol.GetStereoGroups()):
        return "CXSMILES:" + Chem.MolToCXSmiles(mol), inchikey.split("-")[0]
    return inchikey, inchikey.split("-")[0]


def structure_key(smiles):
    return structure_keys(smiles)[0]


@lru_cache(maxsize=200_000)
def is_multifragment(smiles):
    """True if a structure still has several fragments after removing known salts/solvents."""
    mol = Chem.MolFromSmiles(smiles.strip()) if isinstance(smiles, str) and smiles.strip() else None
    if mol is None:
        return False
    return len(Chem.GetMolFrags(_salt_remover.remove(mol))) > 1


@lru_cache(maxsize=200_000)
def _fingerprint(smiles):
    mol = _standardize(smiles)
    return None if mol is None else _morgan.GetFingerprint(mol)


def tanimoto(smiles_a, smiles_b):
    fa, fb = _fingerprint(smiles_a), _fingerprint(smiles_b)
    if fa is None or fb is None:
        return 0.0
    return DataStructs.TanimotoSimilarity(fa, fb)

# ---------------------------------------------------------------------------
# Measurement normalization
# ---------------------------------------------------------------------------

_SUBSCRIPTS = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
_MOLAR_TO_NM = {"pm": 1e-3, "nm": 1.0, "um": 1e3, "mm": 1e6, "m": 1e9}
_TIME_TO_H = {"s": 1 / 3600, "sec": 1 / 3600, "min": 1 / 60, "mins": 1 / 60, "minutes": 1 / 60,
              "h": 1.0, "hr": 1.0, "hrs": 1.0, "hours": 1.0, "d": 24.0, "days": 24.0}
_P_ENDPOINTS = {"pic50": "ic50", "pki": "ki", "pkd": "kd", "pec50": "ec50", "pgi50": "gi50"}
_ENDPOINT_SYNONYMS = {"inhibition": "%inhibition", "%inh": "%inhibition", "inhibition%": "%inhibition"}
_RELATIONS = {"": "=", "=": "=", "==": "=", "~": "=", "≈": "=", "<": "<", "<=": "<", "≤": "<",
              ">": ">", ">=": ">", "≥": ">", "range": "range"}
_FLIP = {"=": "=", "<": ">", ">": "<", "range": "range"}


def _norm_text(s):
    s = str(s or "").translate(_SUBSCRIPTS).casefold()
    return re.sub(r"[^0-9a-z%]+", "", s)


def _norm_endpoint(s):
    """'IC₅₀ (µM)' -> 'ic50': units written inside the endpoint name are dropped."""
    e = _norm_text(re.sub(r"\(.*?\)|\[.*?\]", "", str(s or "")))
    return _ENDPOINT_SYNONYMS.get(e, e)


def _norm_unit(s):
    """
    Canonical unit string: 'µM' -> 'um', 'mL·min⁻¹·kg⁻¹' and 'mL/min/kg' -> 'ml/min/kg'.
    Inverse factors are written after '/', in source order.
    """
    s = str(s or "").replace("µ", "u").replace("μ", "u").casefold()
    s = re.sub(r"\^?\s*[-−]\s*1\b|⁻¹", "§", s)
    num, den = [], []
    for sep, tok, inv in re.findall(r"([/·*.\s]?)\s*([a-z%]+)(§?)", s):
        (den if sep == "/" or inv else num).append(tok)
    return ".".join(num) + "".join("/" + t for t in den)


def _to_canonical(value, unit):
    """(value, unit) -> (unit_class, value in canonical unit)."""
    u = _norm_unit(unit)
    if u in _MOLAR_TO_NM:
        return "molar", value * _MOLAR_TO_NM[u]
    if u in _TIME_TO_H:
        return "time", value * _TIME_TO_H[u]
    return "unit:" + u, value


def normalize_measurement(m):
    """
    -> (endpoint, target, relation, unit_class, condition, values, from_p), or None if the
    value is not numeric. `values` is one number, or (lo, hi) for a range, in a canonical
    unit: molar concentrations in nM, times in hours; p-values (pIC50 etc.) are converted
    to nM with the relation flipped and `from_p` set, so that a pX printed in the paper and
    a molar value converted by the model can still match within rounding. `condition` is
    the test concentration of single-dose endpoints such as % inhibition.
    """
    raw = m.get("value")
    try:
        values = tuple(float(v) for v in raw) if isinstance(raw, (list, tuple)) else (float(raw),)
    except (TypeError, ValueError):
        return None
    if not values or len(values) > 2 or not all(math.isfinite(v) for v in values):
        return None
    endpoint = _norm_endpoint(m.get("endpoint"))
    target = _norm_text(m.get("target")) if MATCH_TARGET else ""
    relation = "range" if len(values) == 2 else _RELATIONS.get(str(m.get("relation") or "").strip(), "?")
    from_p = endpoint in _P_ENDPOINTS and _norm_unit(m.get("unit")) == ""
    if from_p:
        endpoint, unit_class = _P_ENDPOINTS[endpoint], "molar"
        values, relation = tuple(10.0 ** (9.0 - v) for v in values), _FLIP.get(relation, relation)
    else:
        converted = [_to_canonical(v, m.get("unit")) for v in values]
        unit_class, values = converted[0][0], tuple(v for _, v in converted)
    values = tuple(sorted(values))
    condition = ""
    if m.get("conc") is not None:
        try:
            cls, cval = _to_canonical(float(m["conc"]), m.get("conc_unit"))
            condition = f"{cls}:{cval:.4g}"
        except (TypeError, ValueError):
            condition = "?"
    return endpoint, target, relation, unit_class, condition, values, from_p

# ---------------------------------------------------------------------------
# Per-document scoring
# ---------------------------------------------------------------------------

def _norm_label(s):
    """'Compound 12a' / 'cpd. 12a' / '12a' -> '12a'."""
    s = str(s or "").casefold().strip()
    s = re.sub(r"^(compound|cpd|example|ex)\.?\s*", "", s)
    return re.sub(r"[^0-9a-z]+", "", s)


def _capped(items, cap):
    """Items beyond the cap are kept as empty placeholders: they count, but are never valid."""
    items = list(items or [])
    return items[:cap] + [{}] * max(0, len(items) - cap)


def _compound_table(compounds):
    """compound id -> smiles."""
    return {_norm_label(c["id"]): c.get("smiles") for c in compounds if c.get("id") is not None}


def _multiset_tp(gold_keys, pred_keys):
    pool = {}
    for k in pred_keys:
        pool[k] = pool.get(k, 0) + 1
    tp = 0
    for k in gold_keys:
        if pool.get(k, 0) > 0:
            pool[k] -= 1
            tp += 1
    return tp


def _records(measurements, table, by_label):
    """Measurements -> list of (key_tuple, (values, from_p)). Unresolvable records keep key None."""
    out = []
    for m in measurements:
        norm = normalize_measurement(m)
        cid = _norm_label(m.get("compound_id"))
        if by_label:
            ckey = cid or None
        else:
            ckey = structure_key(table[cid]) if cid in table else None
        if norm is None or ckey is None:
            out.append((None, None))
            continue
        *fields, values, from_p = norm
        out.append(((ckey, *fields), (values, from_p)))
    return out


def _close(gold, pred):
    (gv, gp), (pv, pp) = gold, pred
    if len(gv) != len(pv):
        return False
    if gp != pp:   # one side was converted between pX and molar: allow rounding
        return all(g > 0 and p > 0 and abs(math.log10(g) - math.log10(p)) <= LOG_TOL_CONVERTED for g, p in zip(gv, pv))
    return all(math.isclose(g, p, rel_tol=VALUE_REL_TOL, abs_tol=1e-12) for g, p in zip(gv, pv))


def _record_tp(gold_recs, pred_recs):
    """
    One-to-one matching within each record key. Tolerance windows are order-preserving
    intervals on the value line, so a sorted sweep gives the maximum matching.
    """
    groups = {}
    for k, v in gold_recs:
        if k is not None:
            groups.setdefault(k, ([], []))[0].append(v)
    for k, v in pred_recs:
        if k is not None and k in groups:
            groups[k][1].append(v)
    tp = 0
    for gold_vals, pred_vals in groups.values():
        gold_vals.sort()
        pred_vals.sort()
        used = [False] * len(pred_vals)
        for g in gold_vals:
            for i, p in enumerate(pred_vals):
                if not used[i] and _close(g, p):
                    used[i] = True
                    tp += 1
                    break
    return tp


def score_document(gold, pred):
    """Returns a dict of additive counts for one document (micro-averaged later)."""
    g_compounds = list(gold.get("compounds") or [])
    p_compounds = _capped(pred.get("compounds"), MAX_PRED_COMPOUNDS)
    p_measurements = _capped(pred.get("measurements"), MAX_PRED_RECORDS)
    g_table, p_table = _compound_table(g_compounds), _compound_table(p_compounds)
    g_smiles = [c.get("smiles") for c in g_compounds]
    p_smiles = [c.get("smiles") for c in p_compounds]
    g_keys = [structure_keys(s) for s in g_smiles]
    p_keys = [structure_keys(s) for s in p_smiles]
    g_valid = [k for k in g_keys if k[0] is not None]
    p_valid = [k for k in p_keys if k[0] is not None]
    struct_tp = _multiset_tp([k for k, _ in g_valid], [k for k, _ in p_valid])
    # Predictions for unscorable gold structures (e.g. drawings SMILES cannot express)
    # are excluded from precision just as the gold side is excluded from recall.
    n_gold_unscorable = len(g_keys) - len(g_valid)
    n_pred_struct = len(p_keys) - min(n_gold_unscorable, len(p_keys) - struct_tp)
    tanimoto_sum = sum(max((tanimoto(gs, ps) for ps in p_smiles), default=0.0)
                       for gs, gk in zip(g_smiles, g_keys) if gk[0] is not None)
    g_ids = [(_norm_label(c["id"]), k[0]) for c, k in zip(g_compounds, g_keys) if c.get("id") is not None and k[0]]
    p_ids = [(_norm_label(c["id"]), k[0]) for c, k in zip(p_compounds, p_keys) if c.get("id") is not None]
    g_measurements = list(gold.get("measurements") or [])
    g_recs = _records(g_measurements, g_table, by_label=False)
    p_recs = _records(p_measurements, p_table, by_label=False)
    return {
        "doc_id": gold.get("doc_id"),
        "family_id": gold.get("family_id") or gold.get("doc_id"),
        "slice": gold.get("slice") or "all",
        "n_gold_unscorable": n_gold_unscorable,
        "struct_tp": struct_tp,
        "struct_tp_skeleton": _multiset_tp([s for _, s in g_valid], [s for _, s in p_valid]),
        "n_gold_struct": len(g_valid),
        "n_pred_struct": n_pred_struct,
        "n_invalid_struct": len(p_keys) - len(p_valid),
        "n_multifrag_struct": sum(1 for s in p_smiles if is_multifragment(s)),
        "tanimoto_sum": tanimoto_sum,
        "id_link_tp": _multiset_tp(g_ids, p_ids),
        "n_gold_id": len(g_ids),
        "n_pred_id": len(p_ids),
        "record_tp": _record_tp(g_recs, p_recs),
        "record_tp_label": _record_tp(_records(g_measurements, g_table, by_label=True),
                                      _records(p_measurements, p_table, by_label=True)),
        "n_gold_rec": sum(1 for k, _ in g_recs if k is not None),
        "n_pred_rec": len(p_recs),
        "n_orphan_rec": sum(1 for m in p_measurements if _norm_label(m.get("compound_id")) not in p_table),
    }

# ---------------------------------------------------------------------------
# Aggregation, composite score, bootstrap
# ---------------------------------------------------------------------------

_ID_FIELDS = ("doc_id", "family_id", "slice")
SCALARS = ("auto", "composite", "struct_f1", "record_f1")


def _ratio(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return np.divide(a, b, out=np.zeros(np.broadcast(a, b).shape), where=b > 0)


def _f1(tp, n_pred, n_gold):
    p, r = _ratio(tp, n_pred), _ratio(tp, n_gold)
    return _ratio(2 * p * r, p + r), p, r


def resolve_scalar(per_doc, scalar="auto"):
    """'auto' = composite if the gold has any measurement records, else structure F1."""
    assert scalar in SCALARS, scalar
    if scalar != "auto":
        return scalar
    return "composite" if sum(d["n_gold_rec"] for d in per_doc) > 0 else "struct_f1"


def _scalar_from(tot, scalar):
    """Primary scalar from totals (works elementwise on bootstrap arrays)."""
    struct_f1 = _f1(tot["struct_tp"], tot["n_pred_struct"], tot["n_gold_struct"])[0]
    record_f1 = _f1(tot["record_tp"], tot["n_pred_rec"], tot["n_gold_rec"])[0]
    return {"struct_f1": struct_f1, "record_f1": record_f1,
            "composite": W_STRUCT * struct_f1 + W_RECORD * record_f1}[scalar]


def _totals(per_doc):
    tot = {}
    for d in per_doc:
        for k, v in d.items():
            if k not in _ID_FIELDS:
                tot[k] = tot.get(k, 0) + v
    return tot


def aggregate(per_doc, scalar="auto"):
    mode = resolve_scalar(per_doc, scalar) if per_doc else "struct_f1"
    tot = _totals(per_doc)
    get = lambda k: tot.get(k, 0)
    struct_f1, struct_p, struct_r = _f1(get("struct_tp"), get("n_pred_struct"), get("n_gold_struct"))
    rec_f1, rec_p, rec_r = _f1(get("record_tp"), get("n_pred_rec"), get("n_gold_rec"))
    out = {
        "score": _scalar_from({k: get(k) for k in ("struct_tp", "n_pred_struct", "n_gold_struct",
                                                   "record_tp", "n_pred_rec", "n_gold_rec")}, mode),
        "struct_f1": struct_f1,
        "struct_precision": struct_p,
        "struct_recall": struct_r,
        "struct_f1_nostereo": _f1(get("struct_tp_skeleton"), get("n_pred_struct"), get("n_gold_struct"))[0],
        "struct_tanimoto": _ratio(get("tanimoto_sum"), get("n_gold_struct")),
        "invalid_smiles_rate": _ratio(get("n_invalid_struct"), get("n_pred_struct")),
        "multifragment_rate": _ratio(get("n_multifrag_struct"), get("n_pred_struct")),
        "id_link_f1": _f1(get("id_link_tp"), get("n_pred_id"), get("n_gold_id"))[0],
        "record_f1": rec_f1,
        "record_precision": rec_p,
        "record_recall": rec_r,
        "record_f1_by_label": _f1(get("record_tp_label"), get("n_pred_rec"), get("n_gold_rec"))[0],
        "orphan_record_rate": _ratio(get("n_orphan_rec"), get("n_pred_rec")),
    }
    out = {k: float(v) for k, v in out.items()}
    out.update({"scalar": mode, "n_docs": len(per_doc), "n_gold_struct": get("n_gold_struct"),
                "n_gold_unscorable": get("n_gold_unscorable"), "n_gold_rec": get("n_gold_rec")})
    return out


def _cluster_totals(per_doc, cluster_order=None):
    """-> (cluster ids, {field: array of per-cluster totals})."""
    groups = {}
    for d in per_doc:
        groups.setdefault(d.get("family_id") or d["doc_id"], []).append(d)
    order = cluster_order or sorted(groups)
    fields = [k for k in per_doc[0] if k not in _ID_FIELDS]
    return order, {f: np.array([sum(d[f] for d in groups.get(c, [])) for c in order], dtype=float) for f in fields}


def _bootstrap_scores(cols, weights, scalar):
    """Scalar for every bootstrap replicate: weights is (replicates x clusters) resampling counts."""
    return _scalar_from({f: weights @ v for f, v in cols.items()}, scalar)


def _weights(n_clusters):
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    return rng.multinomial(n_clusters, np.full(n_clusters, 1.0 / n_clusters), size=BOOTSTRAP_SAMPLES).astype(float)


def bootstrap_ci(per_doc, alpha=0.05, scalar="auto"):
    """Percentile CI of the primary scalar, resampling patent families / articles (clusters)."""
    mode = resolve_scalar(per_doc, scalar)
    order, cols = _cluster_totals(per_doc)
    scores = _bootstrap_scores(cols, _weights(len(order)), mode)
    return float(np.quantile(scores, alpha / 2)), float(np.quantile(scores, 1 - alpha / 2))


def paired_compare(base, new, alpha=0.10, scalar="auto"):
    """Paired cluster bootstrap over shared doc_ids: delta = score(new) - score(base)."""
    b = {d["doc_id"]: d for d in base}
    n = {d["doc_id"]: d for d in new}
    ids = sorted(set(b) & set(n))
    assert ids, "no shared doc_ids between the two runs"
    base_docs, new_docs = [b[i] for i in ids], [n[i] for i in ids]
    mode = resolve_scalar(base_docs, scalar)
    order, base_cols = _cluster_totals(base_docs)
    _, new_cols = _cluster_totals([dict(d, family_id=b[d["doc_id"]].get("family_id")) for d in new_docs], order)
    delta = float(aggregate(new_docs, mode)["score"] - aggregate(base_docs, mode)["score"])
    w = _weights(len(order))
    deltas = _bootstrap_scores(new_cols, w, mode) - _bootstrap_scores(base_cols, w, mode)
    return {"delta": delta, "ci_lo": float(np.quantile(deltas, alpha / 2)),
            "ci_hi": float(np.quantile(deltas, 1 - alpha / 2)),
            "p_not_better": float(np.mean(deltas <= 0)), "n_shared": len(ids), "n_clusters": len(order),
            "scalar": mode}


def pool_seeds(runs):
    """Several per-doc files of the same recipe (different seeds) -> one per-doc list of summed counts."""
    pooled = {}
    for run in runs:
        for d in run:
            p = pooled.setdefault(d["doc_id"], {k: d[k] for k in _ID_FIELDS if k in d})
            for k, v in d.items():
                if k not in _ID_FIELDS:
                    p[k] = p.get(k, 0) + v
    return list(pooled.values())

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _read_jsonl(path, strict=True):
    """Returns (rows, n_unparseable). Gold must parse; bad prediction lines are counted, not fatal."""
    rows, bad = [], 0
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("not a JSON object")
                rows.append(row)
            except ValueError:
                if strict:
                    raise
                bad += 1
    return rows, bad


def score_files(gold_path, pred_path):
    """Every gold document is scored; a missing or unparseable prediction counts as empty."""
    gold, _ = _read_jsonl(gold_path)
    pred_rows, n_bad = _read_jsonl(pred_path, strict=False)
    pred = {}
    for d in pred_rows:
        pred.setdefault(d.get("doc_id"), d)  # first prediction wins: no best-of-N
    missing = sum(1 for g in gold if g.get("doc_id") not in pred)
    if missing or n_bad:
        print(f"warning: {missing} gold documents without prediction, {n_bad} unparseable lines", file=sys.stderr)
    return [score_document(g, pred.get(g.get("doc_id"), {})) for g in gold], missing


def _check_rdkit():
    if RDKIT_VERSION != RDKIT_PIN and os.environ.get("SCORE_ALLOW_RDKIT_MISMATCH") != "1":
        sys.exit(f"RDKit {RDKIT_VERSION} != pinned {RDKIT_PIN}; scores would not be comparable "
                 f"(set SCORE_ALLOW_RDKIT_MISMATCH=1 to override for exploration only)")


def main():
    parser = argparse.ArgumentParser(description="Fixed scorer for structure + property extraction")
    sub = parser.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score")
    s.add_argument("--gold", required=True)
    s.add_argument("--pred", required=True)
    s.add_argument("--per-doc-out", default=None)
    s.add_argument("--scalar", choices=SCALARS, default="auto")
    c = sub.add_parser("compare")
    c.add_argument("--base", required=True, help="per-doc file(s), comma-separated seeds")
    c.add_argument("--new", required=True, help="per-doc file(s), comma-separated seeds")
    c.add_argument("--scalar", choices=SCALARS, default="auto")
    args = parser.parse_args()
    _check_rdkit()

    if args.cmd == "score":
        per_doc, n_missing = score_files(args.gold, args.pred)
        if args.per_doc_out:
            with open(args.per_doc_out, "w") as f:
                for d in per_doc:
                    f.write(json.dumps(d) + "\n")
        summary = aggregate(per_doc, args.scalar)
        lo, hi = bootstrap_ci(per_doc, scalar=args.scalar)
        print("---")
        for k, v in summary.items():
            print(f"{k + ':':22s}{v:.6f}" if isinstance(v, float) else f"{k + ':':22s}{v}")
        print(f"{'score_ci95:':22s}[{lo:.6f}, {hi:.6f}]")
        print(f"{'missing_pred_docs:':22s}{n_missing}")
        print(f"{'rdkit_version:':22s}{RDKIT_VERSION}")
        slices = sorted({d["slice"] for d in per_doc})
        if len(slices) > 1:
            for name in slices:
                sl = aggregate([d for d in per_doc if d["slice"] == name], summary["scalar"])
                print(f"{'slice/' + name + ':':22s}score={sl['score']:.6f} n_docs={sl['n_docs']}")
    else:
        load = lambda spec: pool_seeds([_read_jsonl(p)[0] for p in spec.split(",")])
        r = paired_compare(load(args.base), load(args.new), scalar=args.scalar)
        print("---")
        print(f"{'delta:':22s}{r['delta']:+.6f}")
        print(f"{'delta_ci90:':22s}[{r['ci_lo']:+.6f}, {r['ci_hi']:+.6f}]")
        print(f"{'p_not_better:':22s}{r['p_not_better']:.4f}")
        print(f"{'n_shared_docs:':22s}{r['n_shared']}")
        print(f"{'n_clusters:':22s}{r['n_clusters']}")
        print(f"{'scalar:':22s}{r['scalar']}")


if __name__ == "__main__":
    main()
