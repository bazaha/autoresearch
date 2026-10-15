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
                     "value": [10, 30], "unit": "uM", "assay_type": "cell"},
                    {"compound_id": "12a", "target": "", "endpoint": "CL", "value": "25",
                     "unit": "mL/min/kg", "species": "rat", "dose": 1, "dose_unit": "mg/kg", "route": "iv"}]}

`family_id` (patent family / article) is the bootstrap cluster and defaults to doc_id;
`slice` (gold only) groups documents for per-slice reporting and slice weighting; `tags`
(gold only, optional list such as ["stereo", "markush", "handwritten"]) marks difficulty
subsets that are reported separately (a document may carry several tags). An OCSR crop is
simply a document with one compound and no measurements. A value given as [lo, hi] is a
range; value strings such as ">10", "10–30", "12 ± 3", "1.2 × 10⁻⁵", "10 µM" (unit inside the
value) and "n.d." are parsed, and a unit missing from `unit` is taken from the value string or
from the endpoint header ("IC50 (µM)", "Papp (10⁻⁶ cm/s)"). Compound units (mL/min/kg vs
L/h/kg, ng·h/mL vs µg·h/mL) are compared by dimension. Record key v2: the optional context
fields assay_type, species, matrix (or cell_line), variant, ph, dose/dose_unit and route --
also recognized in the header ("CLint (HLM)", "logD7.4") -- count only where the gold fills
them. Labels keep Latin, Greek and Chinese characters ("PI3Kα", "抑制率").

Structures are SMILES/CXSMILES. Markush/R-group positions are dummy atoms written as
`[1*]`, `[*:1]` or a CXSMILES label (`*... |$R1$|`, `_R1`); all three mean R1. Racemic or
relative stereo uses CXSMILES enhanced stereo (`|&1:1|`, `|o1:1|`).

Usage:
    python score.py score   --gold gold.jsonl --pred pred.jsonl [--per-doc-out per_doc.jsonl]
    python score.py compare --base seed0.jsonl,seed1.jsonl --new seed0.jsonl,seed1.jsonl
    (compare takes per-doc files written by `score`; with several seeds per side, seed-to-seed
    variance enters the interval through a hierarchical family x seed bootstrap)
    Both accept --scalar and --slice-weights (e.g. inhouse=0.3,biovista=0.3,uspto=0.2,mrbw=0.2);
    compare also takes --alpha (two-sided CI level, default 0.10) and prints paired deltas with
    bootstrap SEs for every slice, tag and guardrail rate.
"""

import argparse
import hashlib
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
# Structures with one of these metals bonded into them (coordination complexes, organometallics)
# are keyed as drawn (standard InChI): salt stripping and RDKit's metal disconnection would
# reduce cisplatin to ammonia. Free metal ions ([Zn+2].[O-]C(=O)...) are ordinary salts.
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
        if any(atom.GetSymbol() in _COMPLEX_METALS and atom.GetDegree() > 0 for atom in mol.GetAtoms()):
            pass  # a metal bonded into the structure: keyed as drawn (free metal ions stay salts)
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
# Atomic units of compound units (clearance, AUC, permeability, doses): token -> (dimension,
# factor to the base unit L, h, g, mol or m). 'mL/min/kg' and 'L/h/kg' both become class
# 'dim:mass-1.time-1.vol+1'; molar concentrations and plain times keep their own classes.
_ATOMIC_UNITS = {
    **{k: ("vol", f) for k, f in {"l": 1.0, "dl": 0.1, "ml": 1e-3, "ul": 1e-6, "nl": 1e-9}.items()},
    **{k: ("time", f) for k, f in _TIME_TO_H.items()},
    **{k: ("mass", f) for k, f in {"kg": 1e3, "g": 1.0, "mg": 1e-3, "ug": 1e-6, "ng": 1e-9, "pg": 1e-12}.items()},
    **{k: ("amount", f) for k, f in {"mol": 1.0, "mmol": 1e-3, "umol": 1e-6, "nmol": 1e-9, "pmol": 1e-12}.items()},
    **{k: ("length", f) for k, f in {"m": 1.0, "cm": 1e-2, "mm": 1e-3, "um": 1e-6, "nm": 1e-9}.items()},
}
_UNIT_ALIASES = {"mpk": "mg/kg", "hr": "h"}
# '×10⁻⁶ cm/s', '10^-6 cm/s', '10-6 cm/s', '1e-6 cm/s': a decimal multiplier written in the unit.
_MULTIPLIER = re.compile(r"^\s*(?:[x×*]\s*)?(?:10\s*(?:\^\s*([+\-−]?\d+)|([⁻⁺]?[⁰¹²³⁴⁵⁶⁷⁸⁹]+)|([\-−]\d+))|1e([+\-]?\d+))\s*", re.I)
_P_ENDPOINTS = {"pic50": "ic50", "pki": "ki", "pkd": "kd", "pec50": "ec50", "pgi50": "gi50"}
_ENDPOINT_SYNONYMS = {
    "inhibition": "%inhibition", "%inh": "%inhibition", "inhibition%": "%inhibition", "%inhibition": "%inhibition",
    "抑制率": "%inhibition", "半数抑制浓度": "ic50",
    "t12": "t1/2", "halflife": "t1/2", "thalf": "t1/2", "半衰期": "t1/2",
    "f%": "f", "%f": "f", "bioavailability": "f", "oralbioavailability": "f", "生物利用度": "f", "口服生物利用度": "f",
    "clearance": "cl", "清除率": "cl", "intrinsicclearance": "clint", "内在清除率": "clint", "vdss": "vss",
    "plasmaproteinbinding": "ppb", "血浆蛋白结合率": "ppb", "表观渗透系数": "papp", "溶解度": "solubility",
}
_PERCENT_ENDPOINTS = {"%inhibition", "f", "ppb"}   # an empty unit means '%'
_RELATIONS = {"": "=", "=": "=", "==": "=", "~": "=", "≈": "=", "<": "<", "<=": "<", "≤": "<",
              ">": ">", ">=": ">", "≥": ">", "range": "range"}
_FLIP = {"=": "=", "<": ">", ">": "<", "range": "range"}
# Latin letters, digits and '%', plus Greek (PI3Kα vs PI3Kβ) and CJK (Chinese headers and targets).
_TEXT_DROP = re.compile(r"[^0-9a-z%Ͱ-Ͽ㐀-鿿豈-﫿]+")


def _norm_text(s):
    return _TEXT_DROP.sub("", str(s or "").translate(_SUBSCRIPTS).casefold())


_AT_CONC = re.compile(r"(?:@|\bat\b)\s*([0-9]*\.?[0-9]+)\s*([a-zA-Zµμ]+(?:/[a-zA-Z]+)?)")
_PH = re.compile(r"(?i)(?:@\s*|\bat\s+)?\bpH\s*=?\s*(\d+(?:\.\d+)?)")
_LOGD_PH = re.compile(r"(?i)\b(log\s*D)\s*(\d+(?:\.\d+)?)\b")
_BRACKETS = re.compile(r"[\(（\[【]\s*([^\)）\]】]*?)\s*[\)）\]】]")
_DIRECTION = re.compile(r"(?i)^\s*[ab]\s*(?:→|->|-|to)\s*[ab]\s*$")


def _endpoint_parts(s):
    """
    'IC₅₀ (µM)' -> ('ic50', {}); 'logD7.4' -> ('logd', {'ph': '7.4'}); 'CLint (HLM)' ->
    ('clint', {'matrix': ..., 'species': 'human'}); 'Papp (A→B)' -> ('pappab', {}).
    Bracketed units, concentrations and pH are removed from the name; bracketed species,
    matrices and transport directions are kept as context or as part of the name.
    """
    s = str(s or "")
    ctx = {}
    ph = _PH.search(s) or _LOGD_PH.search(s)
    if ph:
        ctx["ph"] = f"{float(ph.group(ph.lastindex)):.1f}"
    s = _LOGD_PH.sub(r"\1", _PH.sub("", s))
    s = re.sub(_AT_CONC.pattern, "", s)

    def bracket(mm):
        inner = mm.group(1)
        if _DIRECTION.match(inner):
            return inner
        for field, value in _context_from_text(inner).items():
            ctx.setdefault(field, value)
        return ""

    e = _norm_text(_BRACKETS.sub(bracket, s))
    return _ENDPOINT_SYNONYMS.get(e, e), ctx


def _norm_endpoint(s):
    return _endpoint_parts(s)[0]


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
    u = ".".join(num) + "".join("/" + t for t in den)
    return _UNIT_ALIASES.get(u, u)


def _split_multiplier(unit):
    """'×10⁻⁶ cm/s' -> (1e-6, 'cm/s')."""
    s = str(unit or "")
    m = _MULTIPLIER.match(s)
    if not m or not re.search(r"[a-zA-Zµμ%]", s[m.end():]):
        return 1.0, s
    exp = next(g for g in m.groups() if g is not None).translate(_SUPERSCRIPTS).replace("−", "-")
    return 10.0 ** int(exp), s[m.end():]


def _unit_dims(u):
    """Canonical unit string -> (dimension signature, factor to base units), or None."""
    parts = u.split("/")
    exps, factor = {}, 1.0
    for sign, toks in ((1, parts[0].split(".") if parts[0] else []), (-1, parts[1:])):
        for tok in toks:
            if tok not in _ATOMIC_UNITS:
                return None
            dim, f = _ATOMIC_UNITS[tok]
            exps[dim] = exps.get(dim, 0) + sign
            factor = factor * f if sign > 0 else factor / f
    if not exps:
        return None
    return ".".join(f"{d}{e:+d}" for d, e in sorted(exps.items()) if e) or "ratio", factor


def _is_pow10(f):
    return abs(math.log10(f) - round(math.log10(f))) < 1e-9


def _to_canonical(value, unit, mw=None):
    """
    (value, unit) -> (unit_class, value in canonical unit, converted). `converted` is set when
    the value went through a non-decimal factor (mass -> molar via MW, minutes -> hours,
    mL/min/kg -> L/h/kg ...), so that it is compared against the gold's printed rounding.
    """
    mult, rest = _split_multiplier(unit)
    value = value * mult
    u = _norm_unit(rest)
    if u in _MOLAR_TO_NM:
        return "molar", value * _MOLAR_TO_NM[u], False
    if u in _MASS_TO_G_PER_L and mw:
        return "molar", value * _MASS_TO_G_PER_L[u] / mw * 1e9, True
    if u in _TIME_TO_H:
        return "time", value * _TIME_TO_H[u], not _is_pow10(_TIME_TO_H[u])
    dims = _unit_dims(u)
    if dims:
        return "dim:" + dims[0], value * dims[1], not _is_pow10(dims[1])
    return "unit:" + u, value, False


def _looks_like_unit(text):
    mult, rest = _split_multiplier(text)
    u = _norm_unit(rest)
    return bool(u) and (u in _MOLAR_TO_NM or u in _MASS_TO_G_PER_L or u in _TIME_TO_H or u == "%"
                        or _unit_dims(u) is not None)


# Record context (key v2). A field counts only when the gold fills it ("gold-defined fields"):
# inferring the right species is rewarded, filling in a wrong one is not, leaving it out when
# the gold has it is a miss.
_CONTEXT_FIELDS = ("assay_type", "species", "matrix", "variant", "ph", "dose", "route")
_FIELD_ALIASES = {"cell_line": "matrix", "cell_line_or_matrix": "matrix", "mutant": "variant"}
_SPECIES = {"human": "human", "homosapiens": "human", "人": "human", "rat": "rat", "大鼠": "rat",
            "mouse": "mouse", "mice": "mouse", "小鼠": "mouse", "dog": "dog", "beagle": "dog", "beagledog": "dog",
            "犬": "dog", "比格犬": "dog", "monkey": "monkey", "cyno": "monkey", "cynomolgus": "monkey",
            "cynomolgusmonkey": "monkey", "rhesus": "monkey", "rhesusmonkey": "monkey", "猴": "monkey",
            "食蟹猴": "monkey", "minipig": "minipig", "rabbit": "rabbit", "hamster": "hamster", "guineapig": "guineapig"}
_MATRICES = {"livermicrosomes": "liver_microsomes", "microsomes": "liver_microsomes", "lm": "liver_microsomes",
             "肝微粒体": "liver_microsomes", "hepatocytes": "hepatocytes", "肝细胞": "hepatocytes",
             "plasma": "plasma", "血浆": "plasma", "blood": "blood", "全血": "blood", "serum": "serum", "血清": "serum"}
_MATRIX_SPECIES = {"hlm": ("liver_microsomes", "human"), "rlm": ("liver_microsomes", "rat"),
                   "mlm": ("liver_microsomes", "mouse"), "dlm": ("liver_microsomes", "dog"),
                   "cylm": ("liver_microsomes", "monkey"), "mklm": ("liver_microsomes", "monkey"),
                   "hh": ("hepatocytes", "human"), "rh": ("hepatocytes", "rat")}
_ASSAY_TYPES = {"biochemical": "biochemical", "biochem": "biochemical", "enzymatic": "biochemical",
                "enzyme": "biochemical", "enzymeassay": "biochemical", "生化": "biochemical", "酶": "biochemical",
                "酶活": "biochemical", "cell": "cell", "cellular": "cell", "cellbased": "cell", "细胞": "cell",
                "invivo": "invivo", "体内": "invivo", "binding": "binding", "结合": "binding"}
_ROUTES = {"iv": "iv", "intravenous": "iv", "静脉": "iv", "静脉注射": "iv", "po": "po", "oral": "po",
           "peroral": "po", "口服": "po", "灌胃": "po", "ip": "ip", "intraperitoneal": "ip", "腹腔": "ip",
           "腹腔注射": "ip", "sc": "sc", "subcutaneous": "sc", "皮下": "sc", "皮下注射": "sc"}
_WILD_TYPE = {"wt", "wildtype", "野生型"}


def _matrix_species(text):
    """'HLM' / 'human liver microsomes' / '人肝微粒体' -> (matrix, species or None)."""
    t = _norm_text(text)
    if t in _MATRIX_SPECIES:
        return _MATRIX_SPECIES[t]
    if t in _MATRICES:
        return _MATRICES[t], None
    for word, species in sorted(_SPECIES.items(), key=lambda kv: -len(kv[0])):
        if t.startswith(word) and t[len(word):] in _MATRICES:
            return _MATRICES[t[len(word):]], species
    return t, None


def _context_from_text(text):
    """Context recognizable in a bracket of an endpoint header: '(HLM)', '(rat)', '(cell)'."""
    t = _norm_text(text)
    if not t or _looks_like_unit(text):
        return {}
    if t in _SPECIES:
        return {"species": _SPECIES[t]}
    if t in _ASSAY_TYPES:
        return {"assay_type": _ASSAY_TYPES[t]}
    if t in _ROUTES:
        return {"route": _ROUTES[t]}
    matrix, species = _matrix_species(text)
    if matrix != t or species:
        return {"matrix": matrix, **({"species": species} if species else {})}
    return {}


def _context(m, from_endpoint):
    """Normalized context fields of a measurement -> tuple of (field, value) pairs."""
    raw = dict(m)
    for alias, field in _FIELD_ALIASES.items():
        if raw.get(field) in (None, "") and raw.get(alias) not in (None, ""):
            raw[field] = raw[alias]
    ctx = {}
    for field in _CONTEXT_FIELDS:
        v = raw.get(field)
        if v is None or str(v).strip() == "":
            continue
        if field == "species":
            t = _norm_text(v)
            ctx[field] = _SPECIES.get(t, t)
        elif field == "matrix":
            matrix, species = _matrix_species(v)
            ctx[field] = matrix
            if species:
                ctx.setdefault("species", species)
        elif field == "assay_type":
            t = _norm_text(v)
            ctx[field] = _ASSAY_TYPES.get(t, t)
        elif field == "route":
            t = _norm_text(v)
            ctx[field] = _ROUTES.get(t, t)
        elif field == "variant":
            toks = sorted(_norm_text(x) for x in re.split(r"[/,;+&\s]+", str(v)) if _norm_text(x))
            ctx[field] = "wt" if set(toks) <= _WILD_TYPE else "|".join(toks)
        elif field == "ph":
            try:
                ctx[field] = f"{float(v):.1f}"
            except (TypeError, ValueError):
                ctx[field] = _norm_text(v)
        elif field == "dose":
            try:
                cls, val, _ = _to_canonical(float(v), raw.get("dose_unit"))
                ctx[field] = f"{cls}:{val:.4g}"
            except (TypeError, ValueError):
                ctx[field] = _norm_text(v)
    for field, value in from_endpoint.items():   # context written in the endpoint header
        ctx.setdefault(field, value)
    if ctx.get("species") and ctx.get("species") not in _SPECIES.values():
        ctx["species"] = _norm_text(ctx["species"])
    return tuple(sorted((k, v) for k, v in ctx.items() if v))


_NUM = r"[+-]?(?:\d+\.?\d*|\.\d+)(?:e[+-]?\d+)?"
_RANGE = re.compile(rf"^({_NUM})\s*(?:-|–|—|to)\s*({_NUM})$", re.I)
_PREFIX = re.compile(r"^(<=|>=|≤|≥|<|>|~|≈|=)\s*")
_WITH_UNIT = re.compile(rf"^({_NUM}(?:\s*(?:-|–|—|to)\s*{_NUM})?)\s*([^\d\s.+-].*)$", re.I)
_TIMES_TEN = re.compile(r"\s*[x×*·]\s*10\s*\^?\s*([⁻⁺+-]?[0-9⁰¹²³⁴⁵⁶⁷⁸⁹]+)")
_SUPERSCRIPTS = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺", "0123456789-+")
_PLUS_MINUS = re.compile(rf"\s*\(?\s*(?:±|\+/-|\+-)\s*{_NUM}\s*\)?|\s*\(\s*n\s*=\s*\d+\s*\)", re.I)


def _header_unit(endpoint):
    """Unit written in the column header / endpoint name, e.g. 'IC50 (µM)' -> 'µM'."""
    for group in _BRACKETS.findall(str(endpoint or "")):
        if _looks_like_unit(group):
            return group
    return ""


def _half_unit(text):
    """Half a unit in the last printed digit: '7.26' -> 0.005, '50' -> 0.5, '1.2e-5' -> 5e-7."""
    try:
        return 0.5 * 10.0 ** Decimal(text).as_tuple().exponent
    except (InvalidOperation, TypeError, ValueError):
        return 0.0


def _parse_value(raw, relation):
    """
    -> (numbers, half_units, relation, category, inline_unit). Accepts numbers, [lo, hi], and
    strings such as '1,200', '>10', '10-30', '10 to 30', '12 ± 3' (mean kept), '1.2 × 10⁻⁵',
    '10 µM' (unit written in the value); half_units is the printed precision of each number
    (store gold values as printed strings to keep trailing zeros). Anything else non-empty
    ('NT', 'inactive', a binned 'A' or '+++') is a categorical value compared as text.
    """
    if isinstance(raw, bool) or raw is None:
        return None
    items = list(raw) if isinstance(raw, (list, tuple)) else [raw]
    if isinstance(raw, (list, tuple)) or isinstance(raw, (int, float)):
        try:
            return (tuple(float(v) for v in items), tuple(_half_unit(repr(v) if isinstance(v, float) else str(v)) for v in items),
                    relation, None, "")
        except (TypeError, ValueError):
            return None
    s = str(raw).strip().replace("−", "-").replace(",", "")
    s = _TIMES_TEN.sub(lambda mm: "e" + mm.group(1).translate(_SUPERSCRIPTS), s)
    s = _PLUS_MINUS.sub("", s).strip()
    m = _PREFIX.match(s)
    if m:
        relation, s = (m.group(1) if relation in ("", "=") else relation), s[m.end():]
    unit = ""
    wu = None if _RANGE.match(s) or re.fullmatch(_NUM, s, re.I) else _WITH_UNIT.match(s)
    if wu and re.search(r"[a-zA-Zµμ%]", wu.group(2)):
        s, unit = wu.group(1).strip(), wu.group(2).strip()
    r = _RANGE.match(s)
    if r:
        return (float(r.group(1)), float(r.group(2))), (_half_unit(r.group(1)), _half_unit(r.group(2))), relation, None, unit
    try:
        return (float(s),), (_half_unit(s),), relation, None, unit
    except ValueError:
        token = re.sub(r"[^0-9a-z+\-]+", "", (s + unit).casefold())
        return ((), (), relation, token, "") if token else None


def normalize_measurement(m, mw=None):
    """
    -> (endpoint, target, relation, unit_class, condition, values, converted, bounds, context),
    or None if there is no value, or if a non-empty endpoint or target normalizes to nothing.
    `bounds` is the printed-rounding interval of each value, in the same canonical unit.
    `values` is one number, (lo, hi) for a range, or () for a categorical value (whose text
    then sits in `relation`), in a canonical unit: molar concentrations in nM, times in hours,
    compound units (mL/min/kg, ng·h/mL, 10⁻⁶ cm/s ...) in base units. p-values (pIC50 etc.)
    are converted to nM with the relation flipped, and mass concentrations (µg/mL) to nM using
    `mw`, the record's structure; `converted` marks non-decimal conversions, so that the
    printed and a converted form can still match within rounding. `condition` is the test
    concentration of single-dose endpoints such as % inhibition, given as conc/conc_unit or
    written in the endpoint ('% inh @ 1 µM'). An empty unit field falls back to a unit written
    in the value ('10 µM') or in the header ('IC50 (µM)', 'Papp (10⁻⁶ cm/s)'). `context` holds
    the record-key-v2 fields (assay_type, species, matrix, variant, ph, dose, route), also
    when written in the header ('CLint (HLM)', 'logD7.4'); they count only where the gold has them.
    """
    parsed = _parse_value(m.get("value"), str(m.get("relation") or "").strip())
    if parsed is None:
        return None
    values, halves, relation, category, inline_unit = parsed
    if len(values) > 2 or not all(math.isfinite(v) for v in values):
        return None
    endpoint, header_ctx = _endpoint_parts(m.get("endpoint"))
    target = _norm_text(m.get("target")) if MATCH_TARGET else ""
    if (str(m.get("endpoint") or "").strip() and not endpoint) or \
            (MATCH_TARGET and str(m.get("target") or "").strip() and not target):
        return None   # e.g. a label in a script this scorer does not keep: fail closed
    unit = m.get("unit")
    if not str(unit or "").strip():   # unit written in the value, else in the column header
        unit = inline_unit or _header_unit(m.get("endpoint")) or ("%" if endpoint in _PERCENT_ENDPOINTS else "")
    bounds = tuple((v - h, v + h) for v, h in zip(values, halves))
    relation = "range" if len(values) == 2 else _RELATIONS.get(relation, "?")
    converted = False
    if category is not None:
        relation, unit_class = "cat:" + category, ""
    elif endpoint in _P_ENDPOINTS and _norm_unit(unit) in _PX_UNITS:
        endpoint, unit_class, converted = _P_ENDPOINTS[endpoint], "molar", True
        values, relation = tuple(10.0 ** (9.0 - v) for v in values), _FLIP.get(relation, relation)
        bounds = tuple((10.0 ** (9.0 - hi), 10.0 ** (9.0 - lo)) for lo, hi in bounds)
    else:
        canon = [_to_canonical(v, unit, mw) for v in values]
        unit_class, converted = canon[0][0], canon[0][2]
        scale = [c[1] / v if v else _to_canonical(1.0, unit, mw)[1] for c, v in zip(canon, values)]
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
    return endpoint, target, relation, unit_class, condition, values, converted, bounds, _context(m, header_ctx)

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
    """Measurements -> list of (key_tuple, (values, converted, bounds, context)). Unresolvable records keep key None."""
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
        *fields, values, converted, bounds, context = norm
        out.append(((ckey, *fields), (values, converted, bounds, context)))
    return out


def _close(gold, pred):
    """Values agree within tolerance, and every context field the gold fills agrees."""
    (gv, gc, gb, g_ctx), (pv, pc, _, p_ctx) = gold, pred
    if len(gv) != len(pv) or not set(g_ctx) <= set(p_ctx):
        return False
    if gc != pc:   # one side was converted: the prediction must round to the gold as printed
        return all(lo * (1 - 1e-9) <= p <= hi * (1 + 1e-9) for (lo, hi), p in zip(gb, pv))
    return all(math.isclose(g, p, rel_tol=VALUE_REL_TOL, abs_tol=1e-12) for g, p in zip(gv, pv))


def _max_matching(gold_vals, pred_vals):
    """Maximum bipartite matching (augmenting paths) between gold and predicted values."""
    owner = [-1] * len(pred_vals)

    def assign(g, seen):
        for j, p in enumerate(pred_vals):
            if j not in seen and _close(gold_vals[g], p):
                seen.add(j)
                if owner[j] < 0 or assign(owner[j], seen):
                    owner[j] = g
                    return True
        return False

    return sum(assign(g, set()) for g in range(len(gold_vals)))


def _record_tp(gold_recs, pred_recs):
    """One-to-one maximum matching of values within each record key."""
    groups = {}
    for k, v in gold_recs:
        if k is not None:
            groups.setdefault(k, ([], []))[0].append(v)
    for k, v in pred_recs:
        if k is not None and k in groups:
            groups[k][1].append(v)
    return sum(_max_matching(g, p) for g, p in groups.values())


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
    record_tp = _record_tp(g_recs, p_recs)
    # as for structures: predictions of gold records this scorer cannot represent are excluded
    n_gold_rec_unscorable = sum(1 for k, _ in g_recs if k is None)
    n_pred_rec = len(p_recs) - min(n_gold_rec_unscorable, len(p_recs) - record_tp)
    return {
        "doc_id": gold.get("doc_id"),
        "family_id": gold.get("family_id") or gold.get("doc_id"),
        "slice": gold.get("slice") or "all",
        "tags": sorted(set(gold.get("tags") or [])),
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
        "record_tp": record_tp,
        "record_tp_label": _record_tp(_records(g_measurements, g_table, by_label=True),
                                      _records(p_measurements, p_table, by_label=True)),
        "n_gold_rec": sum(1 for k, _ in g_recs if k is not None),
        "n_gold_rec_unscorable": n_gold_rec_unscorable,
        "n_pred_rec": n_pred_rec,
        "n_orphan_rec": sum(1 for m in p_measurements if _norm_label(m.get("compound_id")) not in p_table),
    }

# ---------------------------------------------------------------------------
# Aggregation, composite score, bootstrap
# ---------------------------------------------------------------------------

_ID_FIELDS = ("doc_id", "family_id", "slice", "tags")
SCALARS = ("auto", "composite", "struct_f1", "record_f1")
# Guardrail rates reported with paired deltas and bootstrap SEs by `compare` (numerator, denominator).
GUARDRAILS = {"invalid_smiles_rate": ("n_invalid_struct", "n_pred_struct"),
              "multifragment_rate": ("n_multifrag_struct", "n_pred_struct"),
              "orphan_record_rate": ("n_orphan_rec", "n_pred_rec")}


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


def parse_slice_weights(spec):
    """'inhouse=0.3,biovista=0.3,uspto=0.2,mrbw=0.2' -> dict, or None for a plain micro-average."""
    if not spec:
        return None
    return {name.strip(): float(w) for name, w in (part.split("=") for part in spec.split(","))}


def _scalar_from(tot, scalar):
    """Primary scalar from totals (works elementwise on bootstrap arrays)."""
    struct_f1 = _f1(tot["struct_tp"], tot["n_pred_struct"], tot["n_gold_struct"])[0]
    record_f1 = _f1(tot["record_tp"], tot["n_pred_rec"], tot["n_gold_rec"])[0]
    return {"struct_f1": struct_f1, "record_f1": record_f1,
            "composite": W_STRUCT * struct_f1 + W_RECORD * record_f1}[scalar]


def _primary(by_slice, scalar, slice_weights):
    """
    Primary scalar from per-slice totals. Without weights: the micro-average over everything.
    With weights: sum_s w_s * scalar_s over the weighted slices present, renormalized
    (slices without a weight are reported but do not enter the primary scalar).
    """
    if not slice_weights:
        fields = next(iter(by_slice.values()))
        return _scalar_from({f: sum(t[f] for t in by_slice.values()) for f in fields}, scalar)
    present = [s for s in slice_weights if s in by_slice]
    assert present, f"none of the weighted slices {sorted(slice_weights)} is in the data"
    total = sum(slice_weights[s] for s in present)
    return sum(slice_weights[s] * _scalar_from(by_slice[s], scalar) for s in present) / total


def _totals(per_doc):
    tot = {}
    for d in per_doc:
        for k, v in d.items():
            if k not in _ID_FIELDS:
                tot[k] = tot.get(k, 0) + v
    return tot


def _by_slice(per_doc):
    groups = {}
    for d in per_doc:
        groups.setdefault(d.get("slice") or "all", []).append(d)
    return {s: _totals(docs) for s, docs in groups.items()}


def aggregate(per_doc, scalar="auto", slice_weights=None):
    mode = resolve_scalar(per_doc, scalar) if per_doc else "struct_f1"
    tot = _totals(per_doc)
    get = lambda k: tot.get(k, 0)
    struct_f1, struct_p, struct_r = _f1(get("struct_tp"), get("n_pred_struct"), get("n_gold_struct"))
    rec_f1, rec_p, rec_r = _f1(get("record_tp"), get("n_pred_rec"), get("n_gold_rec"))
    out = {
        "score": _primary(_by_slice(per_doc), mode, slice_weights) if per_doc else 0.0,
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
    out.update({"scalar": mode + ("/slice-weighted" if slice_weights else ""), "n_docs": len(per_doc),
                "n_gold_struct": get("n_gold_struct"), "n_gold_unscorable": get("n_gold_unscorable"),
                "n_gold_rec": get("n_gold_rec"), "n_gold_rec_unscorable": get("n_gold_rec_unscorable")})
    return out

# Bootstrap. Clusters are (slice, patent family / article); resampling is stratified within
# slices so every replicate keeps the slice composition fixed. With several seed runs per
# side, seeds are resampled too (hierarchical), so seed-to-seed variance enters the interval.


def _cluster_matrix(per_doc, order=None):
    """-> (cluster keys, {field: per-cluster totals}) for one run."""
    fields = [k for k in per_doc[0] if k not in _ID_FIELDS]
    keys = [(d.get("slice") or "all", d.get("family_id") or d["doc_id"]) for d in per_doc]
    order = order or sorted(set(keys))
    index = {c: i for i, c in enumerate(order)}
    totals = np.zeros((len(order), len(fields)))
    np.add.at(totals, np.array([index[k] for k in keys]),
              np.array([[d.get(f, 0) for f in fields] for d in per_doc], dtype=float))
    return order, {f: totals[:, j] for j, f in enumerate(fields)}


def _stack_runs(runs, order):
    """Seed runs -> {field: (seeds x clusters) array}."""
    per_run = [_cluster_matrix(r, order)[1] for r in runs]
    return {f: np.stack([t[f] for t in per_run]) for f in per_run[0]}


def _stratified_weights(order, rng):
    """(replicates x clusters) resampling counts, multinomial within each slice."""
    w = np.zeros((BOOTSTRAP_SAMPLES, len(order)))
    for s in sorted({k[0] for k in order}):
        idx = [i for i, k in enumerate(order) if k[0] == s]
        w[:, idx] = rng.multinomial(len(idx), np.full(len(idx), 1.0 / len(idx)), size=BOOTSTRAP_SAMPLES)
    return w


def _seed_weights(n_seeds, rng):
    return rng.multinomial(n_seeds, np.full(n_seeds, 1.0 / n_seeds), size=BOOTSTRAP_SAMPLES).astype(float)


def _slice_totals(mats, order, w=None, k=None):
    """
    {slice: {field: totals}}: point totals (w is None), or one value per replicate where w are
    cluster resampling counts and k are seed resampling counts (replicates x seeds).
    """
    out = {}
    for s in sorted({key[0] for key in order}):
        idx = np.array([i for i, key in enumerate(order) if key[0] == s])
        if w is None:
            out[s] = {f: float(m[:, idx].sum()) for f, m in mats.items()}
        else:
            out[s] = {f: ((w[:, idx] @ m[:, idx].T) * k).sum(axis=1) for f, m in mats.items()}
    return out


def bootstrap_ci(per_doc, alpha=0.05, scalar="auto", slice_weights=None):
    """Percentile CI of the primary scalar, resampling (slice, family) clusters within slices."""
    mode = resolve_scalar(per_doc, scalar)
    order, cols = _cluster_matrix(per_doc)
    mats = {f: v[None, :] for f, v in cols.items()}
    w = _stratified_weights(order, np.random.default_rng(BOOTSTRAP_SEED))
    scores = _primary(_slice_totals(mats, order, w, np.ones((BOOTSTRAP_SAMPLES, 1))), mode, slice_weights)
    return float(np.quantile(scores, alpha / 2)), float(np.quantile(scores, 1 - alpha / 2))


def _as_runs(x):
    """A per-doc list (one seed) or a list of per-doc lists (several seeds) -> list of runs."""
    return [x] if x and isinstance(x[0], dict) else list(x)


def _guard_value(by_slice, num, den):
    return _ratio(sum(t[num] for t in by_slice.values()), sum(t[den] for t in by_slice.values()))


def paired_compare(base, new, alpha=0.10, scalar="auto", slice_weights=None):
    """
    delta = score(new) - score(base), paired on shared doc_ids. `base` and `new` are one run
    or a list of seed runs each. Every bootstrap replicate resamples (slice, family) clusters
    within slices (shared by both sides, which is the pairing) and, separately for each side,
    its seeds; with one seed per side, seed variance cannot enter and the result is only a
    screen. Also returns paired deltas with bootstrap SEs for every slice scalar and guardrail
    rate, and for every difficulty tag (gold `tags`, several per document), so that vetoes can
    be scaled to their own noise (veto if delta < -k * se).
    """
    base_runs, new_runs = _as_runs(base), _as_runs(new)
    gold_keys = {d["doc_id"]: (d.get("slice") or "all", d.get("family_id") or d["doc_id"]) for d in base_runs[0]}
    ids = set(gold_keys)
    for run in base_runs + new_runs:
        ids &= {d["doc_id"] for d in run}
    assert ids, "no shared doc_ids between the runs"
    keep = lambda run: [dict(d, slice=gold_keys[d["doc_id"]][0], family_id=gold_keys[d["doc_id"]][1])
                        for d in run if d["doc_id"] in ids]
    base_runs, new_runs = [keep(r) for r in base_runs], [keep(r) for r in new_runs]
    mode = resolve_scalar(base_runs[0], scalar)
    order, _ = _cluster_matrix(base_runs[0])
    base_m, new_m = _stack_runs(base_runs, order), _stack_runs(new_runs, order)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    w = _stratified_weights(order, rng)
    base_k, new_k = _seed_weights(len(base_runs), rng), _seed_weights(len(new_runs), rng)
    base_rep = _slice_totals(base_m, order, w, base_k)
    new_rep = _slice_totals(new_m, order, w, new_k)
    base_pt, new_pt = _slice_totals(base_m, order), _slice_totals(new_m, order)

    deltas = _primary(new_rep, mode, slice_weights) - _primary(base_rep, mode, slice_weights)
    paired = lambda f: (float(f(new_pt) - f(base_pt)), float(np.std(f(new_rep) - f(base_rep), ddof=1)))
    guards = {g: paired(lambda t, num=num, den=den: _guard_value(t, num, den)) for g, (num, den) in GUARDRAILS.items()}
    slices = {s: paired(lambda t, s=s: _scalar_from(t[s], mode)) for s in base_pt}
    tags = {}
    tag_docs = {}  # tag -> doc_ids, from the first base run so every run is cut the same way
    for d in base_runs[0]:
        for t in d.get("tags") or []:
            tag_docs.setdefault(t, set()).add(d["doc_id"])
    for tag, members in sorted(tag_docs.items()):
        sub = lambda runs: _stack_runs([[d for d in r if d["doc_id"] in members] for r in runs], order)
        tb, tn = sub(base_runs), sub(new_runs)
        whole = lambda tot: _scalar_from({f: sum(t[f] for t in tot.values()) for f in next(iter(tot.values()))}, mode)
        pt = whole(_slice_totals(tn, order)) - whole(_slice_totals(tb, order))
        rep_d = whole(_slice_totals(tn, order, w, new_k)) - whole(_slice_totals(tb, order, w, base_k))
        tags[tag] = (float(pt), float(np.std(rep_d, ddof=1)))
    seed_scores = lambda runs: [round(aggregate(r, mode, slice_weights)["score"], 6) for r in runs]
    return {"delta": float(_primary(new_pt, mode, slice_weights) - _primary(base_pt, mode, slice_weights)),
            "ci_lo": float(np.quantile(deltas, alpha / 2)), "ci_hi": float(np.quantile(deltas, 1 - alpha / 2)),
            "p_not_better": float(np.mean(deltas <= 0)), "n_shared": len(ids), "n_clusters": len(order),
            "scalar": mode + ("/slice-weighted" if slice_weights else ""),
            "base_seed_scores": seed_scores(base_runs), "new_seed_scores": seed_scores(new_runs),
            "guardrails": guards, "slices": slices, "tags": tags}


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


def _sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


def main():
    parser = argparse.ArgumentParser(description="Fixed scorer for structure + property extraction")
    sub = parser.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score")
    s.add_argument("--gold", required=True)
    s.add_argument("--pred", required=True)
    s.add_argument("--per-doc-out", default=None)
    s.add_argument("--scalar", choices=SCALARS, default="auto")
    s.add_argument("--slice-weights", default=None, help="e.g. inhouse=0.3,biovista=0.3,uspto=0.2,mrbw=0.2")
    c = sub.add_parser("compare")
    c.add_argument("--base", required=True, help="per-doc file(s), comma-separated seeds")
    c.add_argument("--new", required=True, help="per-doc file(s), comma-separated seeds")
    c.add_argument("--scalar", choices=SCALARS, default="auto")
    c.add_argument("--slice-weights", default=None)
    c.add_argument("--alpha", type=float, default=0.10, help="two-sided; 0.05 gives a 95%% CI")
    args = parser.parse_args()
    _check_rdkit()
    weights = parse_slice_weights(args.slice_weights)

    if args.cmd == "score":
        per_doc, n_missing = score_files(args.gold, args.pred)
        if args.per_doc_out:
            with open(args.per_doc_out, "w") as f:
                for d in per_doc:
                    f.write(json.dumps(d) + "\n")
        summary = aggregate(per_doc, args.scalar, weights)
        lo, hi = bootstrap_ci(per_doc, scalar=args.scalar, slice_weights=weights)
        print("---")
        for k, v in summary.items():
            print(f"{k + ':':22s}{v:.6f}" if isinstance(v, float) else f"{k + ':':22s}{v}")
        print(f"{'score_ci95:':22s}[{lo:.6f}, {hi:.6f}]")
        print(f"{'missing_pred_docs:':22s}{n_missing}")
        print(f"{'rdkit_version:':22s}{RDKIT_VERSION}")
        print(f"{'gold_sha256:':22s}{_sha256(args.gold)}")
        print(f"{'scorer_sha256:':22s}{_sha256(os.path.abspath(__file__))}")
        mode = resolve_scalar(per_doc, args.scalar)
        slices = sorted({d["slice"] for d in per_doc})
        groups = [("slice", n, [d for d in per_doc if d["slice"] == n]) for n in slices] if len(slices) > 1 else []
        groups += [("tag", t, [d for d in per_doc if t in d["tags"]]) for t in sorted({t for d in per_doc for t in d["tags"]})]
        for kind, name, docs in groups:
            sl = aggregate(docs, mode)
            print(f"{kind + '/' + name + ':':22s}score={sl['score']:.6f} n_docs={sl['n_docs']}")
    else:
        load = lambda spec: [_read_jsonl(p)[0] for p in spec.split(",")]
        r = paired_compare(load(args.base), load(args.new), alpha=args.alpha, scalar=args.scalar, slice_weights=weights)
        print("---")
        print(f"{'delta:':22s}{r['delta']:+.6f}")
        print(f"{'delta_ci%d:' % round(100 * (1 - args.alpha)):22s}[{r['ci_lo']:+.6f}, {r['ci_hi']:+.6f}]")
        print(f"{'p_not_better:':22s}{r['p_not_better']:.4f}")
        print(f"{'n_shared_docs:':22s}{r['n_shared']}")
        print(f"{'n_clusters:':22s}{r['n_clusters']}")
        print(f"{'scalar:':22s}{r['scalar']}")
        print(f"{'base_seed_scores:':22s}{r['base_seed_scores']}")
        print(f"{'new_seed_scores:':22s}{r['new_seed_scores']}")
        for group in ("guardrails", "slices", "tags"):
            for name, (d, se) in r[group].items():
                print(f"{group[:-1] + '/' + name + ':':22s}delta={d:+.6f} se={se:.6f}")


if __name__ == "__main__":
    main()
