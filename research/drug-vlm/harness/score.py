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
document with one compound and no measurements. Markush/R-group structures may use
dummy atoms (`*`, `[1*]`, CXSMILES R-labels). A value given as [lo, hi] is a range.

Usage:
    python score.py score   --gold gold.jsonl --pred pred.jsonl [--per-doc-out per_doc.jsonl]
    python score.py compare --base per_doc_base.jsonl --new per_doc_new.jsonl
"""

import argparse
import json
import math
import os
import random
import re
import sys
from functools import lru_cache

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
VALUE_REL_TOL = 1e-3    # relative tolerance after unit normalization
STRIP_SALTS = True      # compare parent compounds (largest fragment, neutralized)
MATCH_TARGET = True     # a record must also name the right target
BOOTSTRAP_SAMPLES = 1000
BOOTSTRAP_SEED = 0

# ---------------------------------------------------------------------------
# Structure normalization
# ---------------------------------------------------------------------------

_morgan = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
_uncharger = rdMolStandardize.Uncharger()


@lru_cache(maxsize=200_000)
def _standardize(smiles):
    """SMILES/CXSMILES -> standardized RDKit mol, or None if unparseable."""
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    mol = Chem.MolFromSmiles(smiles.strip())
    if mol is None:
        return None
    try:
        if STRIP_SALTS:
            mol = rdMolStandardize.FragmentParent(mol)
            mol = _uncharger.uncharge(mol)
        else:
            mol = rdMolStandardize.Cleanup(mol)
        # round-trip through CXSMILES so ring info / sanitization state is fresh
        mol = Chem.MolFromSmiles(Chem.MolToCXSmiles(mol))
    except Exception:
        return None
    return mol


@lru_cache(maxsize=200_000)
def structure_key(smiles):
    """
    Identity key for exact-match scoring. Standard InChIKey (stereo-aware) for
    ordinary molecules; canonical CXSMILES for Markush/R-group structures,
    which InChI cannot represent. Returns None for invalid input.
    """
    mol = _standardize(smiles)
    if mol is None:
        return None
    if any(atom.GetAtomicNum() == 0 for atom in mol.GetAtoms()):
        return "CXSMILES:" + Chem.MolToCXSmiles(mol)
    key = Chem.MolToInchiKey(mol)
    return key or "SMILES:" + Chem.MolToSmiles(mol)


def skeleton_key(key):
    """Stereo/isotope-insensitive key (InChIKey connectivity block) for diagnostics."""
    if key is None or "-" not in key or ":" in key:
        return key
    return key.split("-")[0]


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
    -> (endpoint, target, relation, unit_class, condition, values) with values (a tuple of
    one number, or (lo, hi) for a range) in a canonical unit; None if the value is not
    numeric. Molar concentrations go to nM, times to hours, p-values (pIC50 etc.) are
    converted to nM with the relation flipped. `condition` is the test concentration of
    single-dose endpoints such as % inhibition (canonical units, 4 significant figures).
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
    if endpoint in _P_ENDPOINTS and _norm_unit(m.get("unit")) == "":
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
    return endpoint, target, relation, unit_class, condition, values

# ---------------------------------------------------------------------------
# Per-document scoring
# ---------------------------------------------------------------------------

def _norm_label(s):
    """'Compound 12a' / 'cpd. 12a' / '12a' -> '12a'."""
    s = str(s or "").casefold().strip()
    s = re.sub(r"^(compound|cpd|example|ex)\.?\s*", "", s)
    return re.sub(r"[^0-9a-z]+", "", s)


def _compound_table(doc):
    """compound id -> smiles, plus the list of all smiles in the document."""
    table, all_smiles = {}, []
    for c in doc.get("compounds") or []:
        smi = c.get("smiles")
        all_smiles.append(smi)
        if c.get("id") is not None:
            table[_norm_label(c["id"])] = smi
    return table, all_smiles


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


def _records(doc, table, by_label):
    """Measurements -> list of (key_tuple, values). Unresolvable records keep key None."""
    out = []
    for m in doc.get("measurements") or []:
        norm = normalize_measurement(m)
        cid = _norm_label(m.get("compound_id"))
        if by_label:
            ckey = cid or None
        else:
            ckey = structure_key(table[cid]) if cid in table else None
        if norm is None or ckey is None:
            out.append((None, None))
            continue
        *fields, values = norm
        out.append(((ckey, *fields), values))
    return out


def _close(a, b):
    return len(a) == len(b) and all(math.isclose(x, y, rel_tol=VALUE_REL_TOL, abs_tol=1e-12) for x, y in zip(a, b))


def _record_tp(gold_recs, pred_recs):
    """
    Maximum one-to-one matching. Within one key, tolerance windows are order-preserving
    intervals on the value line, so a sorted sweep gives the optimum.
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
    g_table, g_smiles = _compound_table(gold)
    p_table, p_smiles = _compound_table(pred)
    g_keys = [structure_key(s) for s in g_smiles]
    p_keys = [structure_key(s) for s in p_smiles]
    p_valid = [k for k in p_keys if k is not None]
    g_valid = [k for k in g_keys if k is not None]
    struct_tp = _multiset_tp(g_valid, p_valid)
    # Predictions for unscorable gold structures (e.g. drawings SMILES cannot express)
    # are excluded from precision just as the gold side is excluded from recall.
    n_gold_unscorable = len(g_keys) - len(g_valid)
    n_pred_struct = len(p_keys) - min(n_gold_unscorable, len(p_keys) - struct_tp)
    tanimoto_sum = sum(max((tanimoto(gs, ps) for ps in p_smiles), default=0.0)
                       for gs, gk in zip(g_smiles, g_keys) if gk is not None)
    g_recs = _records(gold, g_table, by_label=False)
    p_recs = _records(pred, p_table, by_label=False)
    g_recs_label = _records(gold, g_table, by_label=True)
    p_recs_label = _records(pred, p_table, by_label=True)
    return {
        "doc_id": gold.get("doc_id"),
        "family_id": gold.get("family_id") or gold.get("doc_id"),
        "slice": gold.get("slice") or "all",
        "n_gold_unscorable": n_gold_unscorable,
        "struct_tp": struct_tp,
        "struct_tp_skeleton": _multiset_tp([skeleton_key(k) for k in g_valid], [skeleton_key(k) for k in p_valid]),
        "n_gold_struct": len(g_valid),
        "n_pred_struct": n_pred_struct,
        "n_invalid_struct": len(p_keys) - len(p_valid),
        "tanimoto_sum": tanimoto_sum,
        "record_tp": _record_tp(g_recs, p_recs),
        "record_tp_label": _record_tp(g_recs_label, p_recs_label),
        "n_gold_rec": sum(1 for k, _ in g_recs if k is not None),
        "n_pred_rec": len(p_recs),
        "n_orphan_rec": sum(1 for m in pred.get("measurements") or []
                            if _norm_label(m.get("compound_id")) not in p_table),
    }

# ---------------------------------------------------------------------------
# Aggregation, composite score, bootstrap
# ---------------------------------------------------------------------------

_ID_FIELDS = ("doc_id", "family_id", "slice")


def _f1(tp, n_pred, n_gold):
    p = tp / n_pred if n_pred else 0.0
    r = tp / n_gold if n_gold else 0.0
    return (2 * p * r / (p + r) if p + r else 0.0), p, r


def aggregate(per_doc):
    tot = {}
    for d in per_doc:
        for k, v in d.items():
            if k not in _ID_FIELDS:
                tot[k] = tot.get(k, 0) + v
    get = lambda k: tot.get(k, 0)
    struct_f1, struct_p, struct_r = _f1(get("struct_tp"), get("n_pred_struct"), get("n_gold_struct"))
    skel_f1, _, _ = _f1(get("struct_tp_skeleton"), get("n_pred_struct"), get("n_gold_struct"))
    rec_f1, rec_p, rec_r = _f1(get("record_tp"), get("n_pred_rec"), get("n_gold_rec"))
    rec_label_f1, _, _ = _f1(get("record_tp_label"), get("n_pred_rec"), get("n_gold_rec"))
    has_records = get("n_gold_rec") > 0
    score = W_STRUCT * struct_f1 + W_RECORD * rec_f1 if has_records else struct_f1
    return {
        "score": score,
        "struct_f1": struct_f1,
        "struct_precision": struct_p,
        "struct_recall": struct_r,
        "struct_f1_nostereo": skel_f1,
        "struct_tanimoto": get("tanimoto_sum") / get("n_gold_struct") if get("n_gold_struct") else 0.0,
        "invalid_smiles_rate": get("n_invalid_struct") / get("n_pred_struct") if get("n_pred_struct") else 0.0,
        "record_f1": rec_f1,
        "record_precision": rec_p,
        "record_recall": rec_r,
        "record_f1_by_label": rec_label_f1,
        "orphan_record_rate": get("n_orphan_rec") / get("n_pred_rec") if get("n_pred_rec") else 0.0,
        "n_docs": len(per_doc),
        "n_gold_struct": get("n_gold_struct"),
        "n_gold_unscorable": get("n_gold_unscorable"),
        "n_gold_rec": get("n_gold_rec"),
    }


def _clusters(per_doc):
    groups = {}
    for d in per_doc:
        groups.setdefault(d.get("family_id") or d.get("doc_id"), []).append(d)
    return list(groups.values())


def _percentile_ci(samples, alpha):
    samples = sorted(samples)
    n = len(samples)
    return samples[int(alpha / 2 * n)], samples[min(n - 1, int((1 - alpha / 2) * n))]


def bootstrap_ci(per_doc, alpha=0.05):
    """Percentile CI of the composite score, resampling patent families / articles (clusters)."""
    rng = random.Random(BOOTSTRAP_SEED)
    clusters = _clusters(per_doc)
    scores = []
    for _ in range(BOOTSTRAP_SAMPLES):
        sample = [d for _ in clusters for d in clusters[rng.randrange(len(clusters))]]
        scores.append(aggregate(sample)["score"])
    return _percentile_ci(scores, alpha)


def paired_compare(base, new, alpha=0.10):
    """Paired cluster bootstrap over shared doc_ids: delta = score(new) - score(base)."""
    b = {d["doc_id"]: d for d in base}
    n = {d["doc_id"]: d for d in new}
    ids = sorted(set(b) & set(n))
    assert ids, "no shared doc_ids between the two runs"
    clusters = {}
    for i in ids:
        clusters.setdefault(b[i].get("family_id") or i, []).append(i)
    clusters = [clusters[k] for k in sorted(clusters)]
    delta = aggregate([n[i] for i in ids])["score"] - aggregate([b[i] for i in ids])["score"]
    rng = random.Random(BOOTSTRAP_SEED)
    deltas = []
    for _ in range(BOOTSTRAP_SAMPLES):
        sample = [i for _ in clusters for i in clusters[rng.randrange(len(clusters))]]
        deltas.append(aggregate([n[i] for i in sample])["score"] - aggregate([b[i] for i in sample])["score"])
    lo, hi = _percentile_ci(deltas, alpha)
    p_not_better = sum(1 for d in deltas if d <= 0) / BOOTSTRAP_SAMPLES
    return {"delta": delta, "ci_lo": lo, "ci_hi": hi, "p_not_better": p_not_better,
            "n_shared": len(ids), "n_clusters": len(clusters)}

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
    c = sub.add_parser("compare")
    c.add_argument("--base", required=True)
    c.add_argument("--new", required=True)
    args = parser.parse_args()
    _check_rdkit()

    if args.cmd == "score":
        per_doc, n_missing = score_files(args.gold, args.pred)
        if args.per_doc_out:
            with open(args.per_doc_out, "w") as f:
                for d in per_doc:
                    f.write(json.dumps(d) + "\n")
        summary = aggregate(per_doc)
        lo, hi = bootstrap_ci(per_doc)
        print("---")
        for k, v in summary.items():
            print(f"{k + ':':22s}{v:.6f}" if isinstance(v, float) else f"{k + ':':22s}{v}")
        print(f"{'score_ci95:':22s}[{lo:.6f}, {hi:.6f}]")
        print(f"{'missing_pred_docs:':22s}{n_missing}")
        print(f"{'rdkit_version:':22s}{RDKIT_VERSION}")
        slices = sorted({d["slice"] for d in per_doc})
        if len(slices) > 1:
            for name in slices:
                sl = aggregate([d for d in per_doc if d["slice"] == name])
                print(f"{'slice/' + name + ':':22s}score={sl['score']:.6f} n_docs={sl['n_docs']}")
    else:
        r = paired_compare(_read_jsonl(args.base)[0], _read_jsonl(args.new)[0])
        print("---")
        print(f"{'delta:':22s}{r['delta']:+.6f}")
        print(f"{'delta_ci90:':22s}[{r['ci_lo']:+.6f}, {r['ci_hi']:+.6f}]")
        print(f"{'p_not_better:':22s}{r['p_not_better']:.4f}")
        print(f"{'n_shared_docs:':22s}{r['n_shared']}")
        print(f"{'n_clusters:':22s}{r['n_clusters']}")


if __name__ == "__main__":
    main()
