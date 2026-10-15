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
    (compare takes per-doc files written by `score`; with several seeds per side, seed-to-seed
    variance enters the interval through a hierarchical family x seed bootstrap)
"""

import argparse
import json
import math
import os
import re
import sys
from decimal import Decimal, InvalidOperation
from functools import lru_cache

import numpy as np
from rdkit import Chem, DataStructs, RDLogger, __version__ as RDKIT_VERSION
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog("rdApp.*")

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify — changing these changes the metric)
# ---------------------------------------------------------------------------

RDKIT_PIN = "2026.09.1"  # canonicalization changes between RDKit releases; scores are only comparable on one version
W_STRUCT = 0.5          # weight of structure F1 in the composite score
W_RECORD = 0.5          # weight of record F1 (structure + value correctly linked)
VALUE_REL_TOL = 1e-3    # relative tolerance when gold and prediction use the same scale
# When exactly one side was converted (pX or mass -> molar), the prediction must fall inside
# the rounding interval of the gold value as printed (half a unit in its last digit).
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
# Coordination complexes are keyed as drawn (standard InChI): salt stripping and RDKit's
# metal disconnection would reduce cisplatin to ammonia. Alkali/alkaline-earth ions stay salts.
_COMPLEX_METALS = {
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh",
    "Pd", "Ag", "Cd", "La", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Al", "Ga", "In", "Sn",
    "Tl", "Pb", "Bi", "Sb", "Ce", "Gd", "Lu",
}
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


def _is_markush(mol):
    return any(atom.GetAtomicNum() == 0 for atom in mol.GetAtoms())


@lru_cache(maxsize=200_000)
def _standardize(smiles):
    """
    SMILES/CXSMILES -> standardized RDKit mol, or None if unparseable. Markush structures
    are not fragment-stripped: their floating substituents (CXSMILES m:) are fragments.
    """
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    mol = Chem.MolFromSmiles(smiles.strip())
    if mol is None:
        return None
    try:
        mol = Chem.RWMol(mol)
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 0:   # one spelling for every R-label convention
                atom.SetIsotope(_rlabel_number(atom))
            for prop in _DUMMY_PROPS:      # labels on real atoms are display aliases only
                if atom.HasProp(prop):
                    atom.ClearProp(prop)
            atom.SetAtomMapNum(0)
        mol = mol.GetMol()
        if _is_markush(mol):
            return mol if mol.GetNumHeavyAtoms() <= MAX_HEAVY_ATOMS else None
        if any(atom.GetSymbol() in _COMPLEX_METALS for atom in mol.GetAtoms()):
            pass  # keyed as drawn
        elif STRIP_SALTS:
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


def _markush_smiles(mol, isomeric=True):
    """
    Canonical text for a Markush structure. Position-variation groups (CXSMILES m:, kept by
    RDKit only as bond properties and dropped on output) are encoded explicitly: each
    floating substituent becomes a canonical fragment SMILES, and the core atoms it may
    attach to are tagged with atom-map labels before the core is canonicalized.
    """
    mol = Chem.Mol(mol)
    if not isomeric:
        Chem.RemoveStereochemistry(mol)
    floating = [b for b in mol.GetBonds() if b.HasProp("_MolFileBondEndPts")]
    if not floating:
        return Chem.MolToCXSmiles(mol)
    frag_of = {a: frag for frag in Chem.GetMolFrags(mol) for a in frag}
    groups = []
    for bond in floating:
        frag = frag_of[bond.GetBeginAtomIdx()]
        ends = [int(x) - 1 for x in bond.GetProp("_MolFileBondEndPts").strip("()").split()[1:]]
        groups.append((Chem.MolFragmentToSmiles(mol, atomsToUse=list(frag)), frag, ends))
    labels = {smi: i + 1 for i, smi in enumerate(sorted({smi for smi, _, _ in groups}))}
    core = Chem.RWMol(mol)
    tags = {}
    for smi, _, ends in groups:
        for a in ends:
            tags.setdefault(a, set()).add(labels[smi])
    for a, ls in tags.items():
        core.GetAtomWithIdx(a).SetAtomMapNum(int("".join(str(l) for l in sorted(ls))))
    for a in sorted({a for _, frag, _ in groups for a in frag}, reverse=True):
        core.RemoveAtom(a)
    subs = sorted(smi for smi, _, _ in groups)
    return Chem.MolToCXSmiles(core) + " m:" + ".".join(subs)


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
    if _is_markush(mol):
        return "MARKUSH:" + _markush_smiles(mol), "MARKUSH:" + _markush_smiles(mol, isomeric=False)
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
    """True if a non-Markush structure still has several fragments after removing salts/solvents."""
    mol = Chem.MolFromSmiles(smiles.strip()) if isinstance(smiles, str) and smiles.strip() else None
    if mol is None or _is_markush(mol):
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


@lru_cache(maxsize=200_000)
def mol_weight(smiles):
    """Average MW of the standardized parent (g/mol), used to convert mass concentrations."""
    mol = _standardize(smiles)
    return None if mol is None or _is_markush(mol) else Descriptors.MolWt(mol)

# ---------------------------------------------------------------------------
# Measurement normalization
# ---------------------------------------------------------------------------

_SUBSCRIPTS = str.maketrans({**dict(zip("₀₁₂₃₄₅₆₇₈₉ₐₑₒₓₕₖₗₘₙₚₛₜᵢ", "0123456789aeoxhklmnpsti")), "½": "1/2"})
_MOLAR_TO_NM = {"pm": 1e-3, "nm": 1.0, "um": 1e3, "mm": 1e6, "m": 1e9,
                "pmol/l": 1e-3, "nmol/l": 1.0, "umol/l": 1e3, "mmol/l": 1e6, "mol/l": 1e9}
_MASS_TO_G_PER_L = {"g/l": 1.0, "mg/ml": 1.0, "ug/ml": 1e-3, "ng/ml": 1e-6, "pg/ml": 1e-9,
                    "mg/l": 1e-3, "ug/l": 1e-6, "ng/l": 1e-9}
_PX_UNITS = {"", "m", "log", "logm", "logunit", "logunits"}
_TIME_TO_H = {"s": 1 / 3600, "sec": 1 / 3600, "min": 1 / 60, "mins": 1 / 60, "minutes": 1 / 60,
              "h": 1.0, "hr": 1.0, "hrs": 1.0, "hours": 1.0, "d": 24.0, "days": 24.0}
_P_ENDPOINTS = {"pic50": "ic50", "pki": "ki", "pkd": "kd", "pec50": "ec50", "pgi50": "gi50"}
_ENDPOINT_SYNONYMS = {"inhibition": "%inhibition", "%inh": "%inhibition", "inhibition%": "%inhibition",
                      "%inhibition": "%inhibition", "t12": "t1/2", "halflife": "t1/2", "thalf": "t1/2"}
_RELATIONS = {"": "=", "=": "=", "==": "=", "~": "=", "≈": "=", "<": "<", "<=": "<", "≤": "<",
              ">": ">", ">=": ">", "≥": ">", "range": "range"}
_FLIP = {"=": "=", "<": ">", ">": "<", "range": "range"}


def _norm_text(s):
    s = str(s or "").translate(_SUBSCRIPTS).casefold()
    return re.sub(r"[^0-9a-z%]+", "", s)


_AT_CONC = re.compile(r"(?:@|\bat\b)\s*([0-9]*\.?[0-9]+)\s*([a-zA-Zµμ]+(?:/[a-zA-Z]+)?)")


def _norm_endpoint(s):
    """'IC₅₀ (µM)' -> 'ic50': units written inside the endpoint name are dropped."""
    e = _norm_text(re.sub(r"\(.*?\)|\[.*?\]|" + _AT_CONC.pattern, "", str(s or "")))
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


def _to_canonical(value, unit, mw=None):
    """(value, unit) -> (unit_class, value in canonical unit, converted_via_mw)."""
    u = _norm_unit(unit)
    if u in _MOLAR_TO_NM:
        return "molar", value * _MOLAR_TO_NM[u], False
    if u in _MASS_TO_G_PER_L and mw:
        return "molar", value * _MASS_TO_G_PER_L[u] / mw * 1e9, True
    if u in _TIME_TO_H:
        return "time", value * _TIME_TO_H[u], False
    return "unit:" + u, value, False


_NUM = r"[+-]?(?:\d+\.?\d*|\.\d+)(?:e[+-]?\d+)?"
_RANGE = re.compile(rf"^({_NUM})\s*(?:-|–|—|to)\s*({_NUM})$", re.I)
_PREFIX = re.compile(r"^(<=|>=|≤|≥|<|>|~|≈|=)\s*")


def _half_unit(text):
    """Half a unit in the last printed digit: '7.26' -> 0.005, '50' -> 0.5, '1.2e-5' -> 5e-7."""
    try:
        return 0.5 * 10.0 ** Decimal(text).as_tuple().exponent
    except (InvalidOperation, TypeError, ValueError):
        return 0.0


def _parse_value(raw, relation):
    """
    -> (numbers, half_units, relation, category). Accepts numbers, [lo, hi], and strings such
    as '1,200', '>10', '10-30', '10 to 30'; half_units is the printed precision of each number
    (store gold values as printed strings to keep trailing zeros). Anything else non-empty
    ('NT', 'inactive', a binned 'A' or '+++') is a categorical value compared as text.
    """
    if isinstance(raw, bool) or raw is None:
        return None
    items = list(raw) if isinstance(raw, (list, tuple)) else [raw]
    if isinstance(raw, (list, tuple)) or isinstance(raw, (int, float)):
        try:
            return (tuple(float(v) for v in items), tuple(_half_unit(repr(v) if isinstance(v, float) else str(v)) for v in items),
                    relation, None)
        except (TypeError, ValueError):
            return None
    s = str(raw).strip().replace("−", "-").replace(",", "")
    m = _PREFIX.match(s)
    if m:
        relation, s = (m.group(1) if relation in ("", "=") else relation), s[m.end():]
    r = _RANGE.match(s)
    if r:
        return (float(r.group(1)), float(r.group(2))), (_half_unit(r.group(1)), _half_unit(r.group(2))), relation, None
    try:
        return (float(s),), (_half_unit(s),), relation, None
    except ValueError:
        token = re.sub(r"[^0-9a-z+\-]+", "", s.casefold())
        return ((), (), relation, token) if token else None


def normalize_measurement(m, mw=None):
    """
    -> (endpoint, target, relation, unit_class, condition, values, converted, bounds), or None
    if there is no value. `bounds` is the printed-rounding interval of each value, in the
    same canonical unit. `values` is one number, (lo, hi) for a range, or () for a categorical
    value (whose text then sits in `relation`), in a canonical unit: molar concentrations in
    nM, times in hours. p-values (pIC50 etc.) are converted to nM with the relation flipped,
    and mass concentrations (µg/mL) to nM using `mw`, the record's structure; `converted`
    marks both, so that the printed and a converted form can still match within rounding.
    `condition` is the test concentration of single-dose endpoints such as % inhibition,
    given as conc/conc_unit or written in the endpoint ('% inh @ 1 µM').
    """
    parsed = _parse_value(m.get("value"), str(m.get("relation") or "").strip())
    if parsed is None:
        return None
    values, halves, relation, category = parsed
    if len(values) > 2 or not all(math.isfinite(v) for v in values):
        return None
    bounds = tuple((v - h, v + h) for v, h in zip(values, halves))
    endpoint = _norm_endpoint(m.get("endpoint"))
    target = _norm_text(m.get("target")) if MATCH_TARGET else ""
    relation = "range" if len(values) == 2 else _RELATIONS.get(relation, "?")
    converted = False
    if category is not None:
        relation, unit_class = "cat:" + category, ""
    elif endpoint in _P_ENDPOINTS and _norm_unit(m.get("unit")) in _PX_UNITS:
        endpoint, unit_class, converted = _P_ENDPOINTS[endpoint], "molar", True
        values, relation = tuple(10.0 ** (9.0 - v) for v in values), _FLIP.get(relation, relation)
        bounds = tuple((10.0 ** (9.0 - hi), 10.0 ** (9.0 - lo)) for lo, hi in bounds)
    else:
        canon = [_to_canonical(v, m.get("unit"), mw) for v in values]
        unit_class, converted = canon[0][0], canon[0][2]
        scale = [c[1] / v if v else 1.0 for c, v in zip(canon, values)]
        values = tuple(c[1] for c in canon)
        bounds = tuple((lo * k, hi * k) for (lo, hi), k in zip(bounds, scale))
    order = sorted(range(len(values)), key=lambda i: values[i])
    values, bounds = tuple(values[i] for i in order), tuple(bounds[i] for i in order)
    conc, conc_unit = m.get("conc"), m.get("conc_unit")
    if conc is None:
        at = _AT_CONC.search(str(m.get("endpoint") or ""))
        if at:
            conc, conc_unit = at.group(1), at.group(2)
    condition = ""
    if conc is not None:
        try:
            cls, cval, _ = _to_canonical(float(conc), conc_unit)
            condition = f"{cls}:{cval:.4g}"
        except (TypeError, ValueError):
            condition = "?"
    return endpoint, target, relation, unit_class, condition, values, converted, bounds

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
    """Measurements -> list of (key_tuple, (values, converted, bounds)). Unresolvable records keep key None."""
    out = []
    for m in measurements:
        cid = _norm_label(m.get("compound_id"))
        norm = normalize_measurement(m, mol_weight(table[cid]) if cid in table else None)
        if by_label:
            ckey = cid or None
        else:
            ckey = structure_key(table[cid]) if cid in table else None
        if norm is None or ckey is None:
            out.append((None, None))
            continue
        *fields, values, converted, bounds = norm
        out.append(((ckey, *fields), (values, converted, bounds)))
    return out


def _close(gold, pred):
    (gv, gc, gb), (pv, pc, _) = gold, pred
    if len(gv) != len(pv):
        return False
    if gc != pc:   # one side was converted: the prediction must round to the gold as printed
        return all(lo * (1 - 1e-9) <= p <= hi * (1 + 1e-9) for (lo, hi), p in zip(gb, pv))
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
        "n_gold_rec_unscorable": sum(1 for k, _ in g_recs if k is None),
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
                "n_gold_unscorable": get("n_gold_unscorable"), "n_gold_rec": get("n_gold_rec"),
                "n_gold_rec_unscorable": get("n_gold_rec_unscorable")})
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


def _as_runs(x):
    """A per-doc list (one seed) or a list of per-doc lists (several seeds) -> list of runs."""
    return [x] if x and isinstance(x[0], dict) else list(x)


def paired_compare(base, new, alpha=0.10, scalar="auto"):
    """
    delta = score(new) - score(base), paired on shared doc_ids. `base` and `new` are one run
    or a list of seed runs each. The bootstrap is hierarchical: every replicate resamples
    patent families / articles (shared by both sides, which is the pairing) and, separately
    for each side, its seeds; so seed-to-seed variance is part of the interval. With one
    seed per side it cannot be, and the result is only a screen.
    """
    base_runs, new_runs = _as_runs(base), _as_runs(new)
    fam = {d["doc_id"]: d.get("family_id") or d["doc_id"] for d in base_runs[0]}
    ids = set(fam)
    for run in base_runs + new_runs:
        ids &= {d["doc_id"] for d in run}
    ids = sorted(ids)
    assert ids, "no shared doc_ids between the runs"
    keep = lambda run: [dict(d, family_id=fam[d["doc_id"]]) for d in run if d["doc_id"] in set(ids)]
    base_runs, new_runs = [keep(r) for r in base_runs], [keep(r) for r in new_runs]
    mode = resolve_scalar(base_runs[0], scalar)
    order, _ = _cluster_totals(base_runs[0])
    stack = lambda runs: {f: np.stack([_cluster_totals(r, order)[1][f] for r in runs])
                          for f in _cluster_totals(runs[0], order)[1]}
    base_m, new_m = stack(base_runs), stack(new_runs)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    w = rng.multinomial(len(order), np.full(len(order), 1.0 / len(order)), size=BOOTSTRAP_SAMPLES).astype(float)

    def replicates(mats, n_seeds):
        k = rng.multinomial(n_seeds, np.full(n_seeds, 1.0 / n_seeds), size=BOOTSTRAP_SAMPLES).astype(float)
        return _scalar_from({f: ((w @ m.T) * k).sum(axis=1) for f, m in mats.items()}, mode)

    point = lambda mats: float(_scalar_from({f: m.sum() for f, m in mats.items()}, mode))
    deltas = replicates(new_m, len(new_runs)) - replicates(base_m, len(base_runs))
    seed_scores = lambda runs: [round(aggregate(r, mode)["score"], 6) for r in runs]
    return {"delta": point(new_m) - point(base_m), "ci_lo": float(np.quantile(deltas, alpha / 2)),
            "ci_hi": float(np.quantile(deltas, 1 - alpha / 2)),
            "p_not_better": float(np.mean(deltas <= 0)), "n_shared": len(ids), "n_clusters": len(order),
            "scalar": mode, "base_seed_scores": seed_scores(base_runs), "new_seed_scores": seed_scores(new_runs)}


def pool_seeds(runs):
    """Several per-doc files of the same recipe (different seeds) -> one per-doc list of summed counts.
    For aggregate reporting only: comparisons should pass the runs separately to paired_compare."""
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
        load = lambda spec: [_read_jsonl(p)[0] for p in spec.split(",")]
        r = paired_compare(load(args.base), load(args.new), scalar=args.scalar)
        print("---")
        print(f"{'delta:':22s}{r['delta']:+.6f}")
        print(f"{'delta_ci90:':22s}[{r['ci_lo']:+.6f}, {r['ci_hi']:+.6f}]")
        print(f"{'p_not_better:':22s}{r['p_not_better']:.4f}")
        print(f"{'n_shared_docs:':22s}{r['n_shared']}")
        print(f"{'n_clusters:':22s}{r['n_clusters']}")
        print(f"{'scalar:':22s}{r['scalar']}")
        print(f"{'base_seed_scores:':22s}{r['base_seed_scores']}")
        print(f"{'new_seed_scores:':22s}{r['new_seed_scores']}")


if __name__ == "__main__":
    main()
