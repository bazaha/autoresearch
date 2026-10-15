"""
Fixed scoring harness for the drug-VLM autoresearch loop (reference implementation).
This is the equivalent of `evaluate_bpb` in prepare.py: deterministic, CPU-only,
and READ-ONLY for the agent. The agent edits train.py; this file decides keep/discard.

Gold and predictions are JSONL files, one document (a page, a crop, or a patent) per line:

  {"doc_id": "US1234567B2-p12", "family_id": "US1234567", "slice": "uspto", "doc_type": "patent",
   "compounds":    [{"id": "12a", "smiles": "CC(=O)Nc1ccc(O)cc1"}],
   "measurements": [{"compound_id": "12a", "target": "EGFR", "endpoint": "IC50",
                     "relation": "=", "value": 12.3, "unit": "nM"},
                    {"compound_id": "12a", "target": "EGFR", "endpoint": "% inhibition",
                     "value": 87, "unit": "%", "conc": 1, "conc_unit": "uM"},
                    {"compound_id": "12a", "target": "hERG", "endpoint": "IC50",
                     "value": [10, 30], "unit": "uM", "assay_type": "cell"},
                    {"compound_id": "12a", "target": "", "endpoint": "CL", "value": "25",
                     "unit": "mL/min/kg", "species": "rat", "dose": 1, "dose_unit": "mg/kg", "route": "iv"}]}

`family_id` (patent family / article; a paper and its SI share one) is the bootstrap cluster
and defaults to doc_id; `slice` (gold only) is the source, used for per-slice reporting and
slice weighting; `doc_type` (gold only: patent / paper / si / report) can replace it as the grouping
(--group-field doc_type, the S2 macro-average; gold `office` and `language` work the same way);
`tags` (gold only, optional list such as ["chiral", "abbrev", "large", "lowres"]) marks
difficulty subsets that are reported separately (a document may carry several). An OCSR crop
is simply a document with compounds and no measurements; there a missing answer costs what a
wrong one costs. A value given as [lo, hi] is a range; value strings such as ">10", "10–30",
"12 ± 3", "1.2 × 10⁻⁵", "10 µM" (unit inside the value) and "n.d." are parsed, and a unit
missing from `unit` is taken from the value string or from the endpoint header ("IC50 (µM)",
"Papp (10⁻⁶ cm/s)"). Compound units (mL/min/kg vs L/h/kg, ng·h/mL vs µg·h/mL) are compared
by dimension. Record key v2: the optional context fields assay_type, species, matrix (or
cell_line), variant, ph, dose/dose_unit, route and time/time_unit -- also recognized in the
header ("CLint (HLM)", "logD7.4") -- count only where the gold fills them. Labels keep Latin, Greek and Chinese characters ("PI3Kα", "抑制率").

Structures are SMILES/CXSMILES. Markush/R-group positions are dummy atoms written as
`[1*]`, `[*:1]` or a CXSMILES label (`*... |$R1$|`, `_R1`); all four mean R1. Racemic or
relative stereo uses CXSMILES enhanced stereo (`|&1:1|`, `|o1:1|`).

Usage:
    python score.py score   --gold gold.jsonl --pred pred.jsonl [--per-doc-out per_doc.jsonl]
    python score.py compare --base seed0.jsonl,seed1.jsonl --new seed0.jsonl,seed1.jsonl
    (compare takes per-doc files written by `score`; with several seeds per side, seed-to-seed
    variance enters the interval through a hierarchical family x seed bootstrap)
    Both accept --scalar, --slice-weights (e.g. inhouse=0.3,biovista=0.3,uspto=0.2,mrbw=0.2; a
    slice without a weight is an error unless --allow-unweighted-slices) and --group-field.
    compare also takes --alpha (two-sided CI level, default 0.10) and --allow-subset (pair on
    shared doc_ids although the runs were scored against different gold), and prints paired
    deltas with bootstrap SEs for every slice, tag and guardrail rate (rates: lower is better).
    Documentation: harness/README.md.
"""

import argparse
import hashlib
import json
import math
import os
import re
import sys
import unicodedata
from decimal import Decimal, InvalidOperation
from bisect import bisect_left, bisect_right
from functools import lru_cache

import numpy as np
from rdkit import Chem, DataStructs, RDLogger, __version__ as RDKIT_VERSION
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog("rdApp.*")

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify — changing these changes the metric)
# ---------------------------------------------------------------------------

SCORER_VERSION = "1.8"  # printed with every score; see harness/README.md section 7
RDKIT_PIN = "2026.09.1"  # canonicalization changes between RDKit releases; scores are only comparable on one version
W_STRUCT = 0.5          # weight of structure F1 in the composite score
W_RECORD = 0.5          # weight of record F1 (structure + value correctly linked)
VALUE_REL_TOL = 1e-3    # relative tolerance when gold and prediction use the same scale
# When exactly one side was converted (pX or mass -> molar), the prediction must fall inside
# the rounding interval of the gold value as printed (half a unit in its last digit).
STRIP_SALTS = True      # remove listed counterions/solvents (other fragments stay), neutralize
MATCH_TARGET = True     # a record must also name the right target
MAX_HEAVY_ATOMS = 500   # larger predicted structures are invalid (keeps scoring time bounded)
MAX_SMILES_CHARS = 5000 # longer strings are invalid before RDKit sees them (runaway output can crash it)
MAX_FIELD_CHARS = 300   # longer value/unit/endpoint/target strings make a record invalid (bounds regex time)
MAX_PRED_COMPOUNDS = 5000   # per document; the rest count as invalid predictions
MAX_PRED_RECORDS = 20000    # per document; the rest count as false positives
BOOTSTRAP_SAMPLES = 1000
BOOTSTRAP_SEED = 0

# ---------------------------------------------------------------------------
# Structure normalization
# ---------------------------------------------------------------------------

_morgan = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
_uncharger = rdMolStandardize.Uncharger()
# Structures with one of these metals bonded into them (coordination complexes, organometallics)
# are keyed as drawn (standard InChI): salt stripping and RDKit's metal disconnection would
# reduce cisplatin to ammonia. Free metal ions ([Zn+2].[O-]C(=O)...) are ordinary salts.
_COMPLEX_METALS = {
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh",
    "Pd", "Ag", "Cd", "La", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Al", "Ga", "In", "Sn",
    "Tl", "Pb", "Bi", "Sb", "Ce", "Gd", "Lu",
}
# Salt formers and solvents removed from multi-fragment structures (ChEMBL-pipeline style: remove
# what is on the list, keep everything else). Keeping "the largest fragment" instead would key
# pyrantel pamoate as pamoic acid, so another drug sold as a pamoate would score as correct.
_COUNTERION_SMILES = (
    # acids
    "Cl", "Br", "I", "F", "OS(=O)(=O)O", "OP(=O)(O)O", "O[N+](=O)[O-]", "OC(=O)C(F)(F)F", "OC=O", "CC(=O)O",
    "CS(=O)(=O)O", "CCS(=O)(=O)O", "O=S(=O)(O)c1ccccc1", "Cc1ccc(S(=O)(=O)O)cc1", "O=S(=O)(O)c1ccc2ccccc2c1",
    "O=S(=O)(O)c1cccc2c(S(=O)(=O)O)cccc12", "O=S(=O)(O)CCS(=O)(=O)O", "OCCS(=O)(=O)O",
    "CC1(C)C2CCC1(CS(=O)(=O)O)C(=O)C2", "OC(=O)/C=C\\C(=O)O", "OC(=O)/C=C/C(=O)O", "OC(=O)C(O)C(O)C(=O)O",
    "OC(=O)CC(O)(CC(=O)O)C(=O)O", "OC(=O)CCC(=O)O", "OC(=O)C(=O)O", "OC(=O)CC(=O)O", "OC(=O)CC(O)C(=O)O",
    "CC(O)C(=O)O", "OC(=O)c1ccccc1", "OC(=O)CNC(=O)c1ccccc1", "OCC(O)C(O)C(O)C(O)C(=O)O", "OC(=O)CCCCC(=O)O",
    "OC(=O)c1cc2ccccc2c(Cc2c(O)c(C(=O)O)cc3ccccc23)c1O", "OC(=O)c1ccc2ccccc2c1O", "OC(=O)C(Cl)Cl",
    "CCCCCCCCCCCCCCCCCC(=O)O",
    # bases
    "N", "OCC(N)(CO)CO", "CNCC(O)C(O)C(O)C(O)CO", "CCNCC", "NCCO", "OCCNCCO", "CCN(CC)CC", "C[N+](C)(C)CCO",
    "C1CNCCN1", "c1ccc(CNCCNCc2ccccc2)cc1", "CC(C)(C)N", "NCCCCC(N)C(=O)O", "NC(=N)NCCCC(N)C(=O)O",
    # solvents
    "O", "CO", "CCO", "CC(C)O", "CC(C)=O", "CS(C)=O", "CCOC(C)=O", "CN(C)C=O", "C1COCCO1", "C1CCOC1",
    "ClCCl", "ClC(Cl)Cl", "CC#N", "CCOCC", "c1ccncc1",
)
_METAL_NUMS = {3, 4, 11, 12, 13, 19, 20, *range(21, 32), 37, 38, *range(39, 51), 55, 56, *range(57, 85)}
_ION_ELEMENTS = {"Li", "Na", "K", "Rb", "Cs", "Mg", "Ca", "Sr", "Ba", "Zn", "Al", "Fe", "Cu", "Mn", "Co", "Ni",
                 "Ag", "Bi", "F", "Cl", "Br", "I"}
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
def _skeleton_key(mol):
    try:
        return Chem.MolToInchiKey(_uncharger.uncharge(mol))[:14]
    except Exception:
        return None


_COUNTERION_KEYS = {_skeleton_key(Chem.MolFromSmiles(x)) for x in _COUNTERION_SMILES} - {None}


def _is_counterion(frag):
    if frag.GetNumHeavyAtoms() == 1 and frag.GetAtomWithIdx(0).GetSymbol() in _ION_ELEMENTS:
        return True   # free metal or halide ion
    return _skeleton_key(frag) in _COUNTERION_KEYS


def _strip_counterions(mol):
    """Remove listed salt formers and solvents; keep every other fragment (a mixture stays a mixture)."""
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    if len(frags) == 1:
        return mol
    keep = [f for f in frags if not _is_counterion(f)]
    if not keep:   # the compound is itself a salt former (e.g. sodium citrate): keep its largest part
        keep = [max(frags, key=lambda f: f.GetNumHeavyAtoms())]
    unique = {}
    for f in keep:   # two equivalents of the same drug (zinc aspirinate) are one parent
        unique.setdefault(Chem.MolToSmiles(_uncharger.uncharge(f)), f)
    keep = list(unique.values())
    out = keep[0]
    for f in keep[1:]:
        out = Chem.CombineMols(out, f)
    return out


@lru_cache(maxsize=200_000)
def _standardize(smiles):
    """
    SMILES/CXSMILES -> standardized RDKit mol, or None if unparseable. Markush structures
    are not fragment-stripped: their floating substituents (CXSMILES m:) are fragments.
    """
    if not isinstance(smiles, str) or not smiles.strip() or len(smiles) > MAX_SMILES_CHARS:
        return None
    mol = Chem.MolFromSmiles(smiles.strip())
    if mol is None or mol.GetNumHeavyAtoms() > 2 * MAX_HEAVY_ATOMS:
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
            if (len(Chem.GetMolFrags(mol)) > 1 or any(a.GetFormalCharge() or a.GetAtomicNum() in _METAL_NUMS
                                                      for a in mol.GetAtoms())
                    or any(g.GetGroupType() != Chem.StereoGroupType.STEREO_ABSOLUTE for g in mol.GetStereoGroups())):
                mol = rdMolStandardize.Cleanup(mol)   # metal disconnection, normalization, reionization
            mol = _uncharger.uncharge(_strip_counterions(mol))
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
    tag_sets = sorted({tuple(sorted(ls)) for ls in tags.values()})
    index = {ts: i + 1 for i, ts in enumerate(tag_sets)}   # {1,2} and {12} stay distinct
    for a, ls in tags.items():
        core.GetAtomWithIdx(a).SetAtomMapNum(index[tuple(sorted(ls))])
    for a in sorted({a for _, frag, _ in groups for a in frag}, reverse=True):
        core.RemoveAtom(a)
    subs = sorted(smi for smi, _, _ in groups)
    return (Chem.MolToCXSmiles(core) + " m:" + ".".join(subs) +
            " t:" + ";".join(",".join(map(str, ts)) for ts in tag_sets))


def structure_keys(smiles):
    """
    -> (key, skeleton) used for exact matching, or (None, None) for invalid input.
    key: standard InChIKey (stereo-aware) for ordinary molecules; canonical CXSMILES for
    Markush/R-group structures (InChI cannot represent them) and for racemic/relative
    (enhanced) stereo, which standard InChI does not distinguish from absolute stereo.
    skeleton: the same identity with stereochemistry removed (diagnostics only).
    """
    if not isinstance(smiles, str):
        return None, None
    try:
        return _structure_keys(smiles)
    except Exception:   # a pathological structure is an invalid prediction, never a crash
        return None, None


@lru_cache(maxsize=200_000)
def _structure_keys(smiles):
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
    if not isinstance(smiles, str):
        return False
    mol = _standardize(smiles)   # listed salts and solvents are already removed there
    return mol is not None and not _is_markush(mol) and len(Chem.GetMolFrags(mol)) > 1


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
                "pmolar": 1e-3, "nmolar": 1.0, "umolar": 1e3, "mmolar": 1e6, "molar": 1e9,
                "pmol/l": 1e-3, "nmol/l": 1.0, "umol/l": 1e3, "mmol/l": 1e6, "mol/l": 1e9}
# Molar written inside compound units ('µM·h', 'M-1 s-1'): mol/L, as (factor in mol/L).
_MOLAR_TOKENS = {"pmolar": 1e-12, "nmolar": 1e-9, "umolar": 1e-6, "mmolar": 1e-3, "molar": 1.0}
# Chinese unit words -> symbols; multi-character words first, SI prefixes last.
_CJK_UNITS = [("千克", "kg"), ("公斤", "kg"), ("摩尔", "mol"), ("小时", "h"), ("分钟", "min"), ("克", "g"), ("升", "l"),
              ("秒", "s"), ("天", "d"), ("日", "d"), ("时", "h"), ("分", "min"), ("毫", "m"), ("微", "u"), ("纳", "n"),
              ("皮", "p"), ("每", "/")]
_UNIT_QUALIFIERS = {"protein", "prot", "microsomalprotein", "mgprotein"}
_FRACTION_ENDPOINTS = {"fu", "fup", "fub", "fubrain", "fumic", "fuinc"}   # an empty unit means a fraction
_MASS_TO_G_PER_L = {"g/l": 1.0, "mg/ml": 1.0, "ug/ml": 1e-3, "ng/ml": 1e-6, "pg/ml": 1e-9,
                    "mg/l": 1e-3, "ug/l": 1e-6, "ng/l": 1e-9}
_PX_UNITS = {"", "m", "molar", "log", "logm", "logmolar", "logunit", "logunits"}
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
_P_ENDPOINTS = {"pic50": "ic50", "pki": "ki", "pkd": "kd", "pec50": "ec50", "pgi50": "gi50"}
_ENDPOINT_SYNONYMS = {
    "inhibition": "%inhibition", "%inh": "%inhibition", "inhibition%": "%inhibition", "%inhibition": "%inhibition",
    "inhibitionrate": "%inhibition", "inhibitionratio": "%inhibition", "抑制率": "%inhibition", "阻害率": "%inhibition",
    "半数抑制浓度": "ic50",
    "t12": "t1/2", "halflife": "t1/2", "thalf": "t1/2", "半衰期": "t1/2", "半減期": "t1/2",
    "f%": "f", "%f": "f", "bioavailability": "f", "oralbioavailability": "f", "生物利用度": "f", "口服生物利用度": "f",
    "clearance": "cl", "清除率": "cl", "クリアランス": "cl", "intrinsicclearance": "clint", "内在清除率": "clint",
    "vdss": "vss", "plasmaproteinbinding": "ppb", "血浆蛋白结合率": "ppb", "表观渗透系数": "papp", "溶解度": "solubility",
    "aucinf": "auc0inf", "auclast": "auc0t", "auct": "auc0t",
}
_PERCENT_ENDPOINTS = {"%inhibition", "f", "ppb"}   # an empty unit means '%'
_RELATIONS = {"": "=", "=": "=", "==": "=", "~": "=", "≈": "=", "<": "<", "<=": "<", "≤": "<", "≦": "<", "⩽": "<",
              ">": ">", ">=": ">", "≥": ">", "≧": ">", "⩾": ">", "range": "range",
              "cat": "="}   # deliverable schema: a binned value ('A', '+++') is compared by its text
_FLIP = {"=": "=", "<": ">", ">": "<", "range": "range"}
# Latin letters, digits and '%', plus Greek (PI3Kα vs PI3Kβ), CJK and kana (Chinese and
# Japanese headers, targets, species). Text is NFKC-normalized first (fullwidth 'ＩＣ５０', '％').
_TEXT_DROP = re.compile("[^0-9a-z%Ͱ-Ͽ぀-ヿㇰ-ㇿ㐀-鿿豈-﫿]+")
_GREEK_NAMES = re.compile(r"alpha|beta|gamma|delta|epsilon|kappa")
_GREEK = {"alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ", "epsilon": "ε", "kappa": "κ"}


_SUPER_RUN = re.compile(r"[⁺⁻]?[⁰¹²³⁴⁵⁶⁷⁸⁹]+")
_DASHES = str.maketrans({c: "-" for c in "‐‑‒–—−﹣－"})


def _nfkc(s):
    """NFKC (fullwidth -> ASCII, 'µ' -> 'μ') after turning superscript exponents into '^n'."""
    s = str(s if s is not None else "").translate(_SUBSCRIPTS)
    s = _SUPER_RUN.sub(lambda m: "^" + m.group(0).translate(_SUPERSCRIPTS), s)
    s = re.sub(r"<sup>\s*([^<]*?)\s*</sup>", r"^\1", s, flags=re.I)
    s = re.sub(r"</?su[pb]>", "", s, flags=re.I)
    return unicodedata.normalize("NFKC", s)


def _norm_text(s):
    """Case-, width- and punctuation-insensitive text key; 'PI3K alpha' == 'PI3Kα'."""
    t = _TEXT_DROP.sub("", _nfkc(s).casefold())
    return _GREEK_NAMES.sub(lambda m: _GREEK[m.group(0)], t)


_AT_CONC = re.compile(r"(?:@|\bat\b)\s*([0-9]*\.?[0-9]+)\s*([a-zA-Zµμ]+(?:/[a-zA-Z]+)?)")
_PH = re.compile(r"(?i)(?:@\s*|\bat\s+)?\bpH\s*[=:]?\s*(\d+(?:\.\d+)?)")
_LOGD_PH = re.compile(r"(?i)\b(log\s*D)\s*(\d+(?:\.\d+)?)(?![\d.])")
_BRACKETS = re.compile(r"[\(\[【]\s*([^\(\)\[\]【】]*?)\s*[\)\]】]")
# Transport direction (Caco-2/MDCK): A/AP/apical -> B/BL/basolateral and back; kept in the name.
_DIRECTION = re.compile(r"(?i)(?<![a-z])(a|ap|apical)\s*(?:→|->|-|>|to|至)\s*(b|bl|basolateral)(?![a-z])"
                        r"|(?<![a-z])(b|bl|basolateral)\s*(?:→|->|-|>|to|至)\s*(a|ap|apical)(?![a-z])")
# AUC intervals: AUC(0-∞) = AUC0-inf = AUCinf; AUC(0-t) = AUC0-last = AUClast; AUC(0-24h) = AUC0-24.
_AUC = re.compile(r"(?i)\bAUC\s*[\(\[]?\s*(?:0\s*[-–—~]\s*(∞|inf(?:inity)?|t|last|\d+(?:\.\d+)?)\s*(?:h|hr|hrs|hours)?"
                  r"|(∞|inf(?:inity)?|last|t)(?![a-z]))\s*[\)\]]?")
_STATISTIC = re.compile(r"(?i)^(?:mean\s*)?(?:±|\+/-|\+-)?\s*(?:s\.?d\.?|s\.?e\.?m\.?|s\.?e\.?|sd|sem)$|^n\s*=\s*\d+$|^mean$")
_STAT_INLINE = re.compile(r"(?i)\s*(?:±|\+/-)\s*(?:s\.?d\.?|s\.?e\.?m\.?|s\.?e\.?)\b|\s*\bmean\s*±\s*\w+")
_MUTATION = re.compile(r"(?<![A-Za-z0-9])(?:[A-Z]\d{1,4}(?:[A-Z]|\*|del|ins)|del\d+|ex(?:on)?\s?\d+\s?(?:del|ins)|WT)(?![A-Za-z0-9])")
_QUANTITY = re.compile(r"^\s*(?:@\s*|at\s+)?(\d+(?:\.\d+)?)\s*([^\d\s].*)$", re.I)


def _norm_unit(s):
    """
    Canonical unit string: 'µM' -> 'um', 'mL·min⁻¹·kg⁻¹', 'mL/min/kg' and 'mL/(min·kg)' ->
    'ml/min/kg'. Inverse factors are written after '/', in source order.
    """
    s = _nfkc(s).translate(_DASHES)
    for word, sym in _CJK_UNITS:
        s = s.replace(word, sym)
    s = re.sub(r"(?<![A-Za-z])([pnuµμm]?)M(?![A-Za-z])", lambda m: (m.group(1) or "") + "molar", s)
    s = s.replace("µ", "u").replace("μ", "u").casefold()
    s = re.sub(r"/\s*\(([^)]*)\)", lambda m: "".join("/" + t for t in re.split(r"[·*.\s]+", m.group(1)) if t), s)
    s = re.sub(r"\^?\s*-\s*1(?![0-9])", "§", s)
    num, den = [], []
    for sep, tok, inv in re.findall(r"([/·*.\s]?)\s*([a-z%]+)(§?)", s):
        if tok in _UNIT_QUALIFIERS:
            continue
        (den if sep == "/" or inv else num).append(tok)
    u = ".".join(num) + "".join("/" + t for t in den)
    return _UNIT_ALIASES.get(u, u)


_LEAD_MULT = re.compile(r"^\s*(?:1\s*)?(?:[x×*·]\s*)?(?:10\s*(?:\^\s*\(?\s*([+-]?\d+)\s*\)?|(-\d+))|1?e([+-]?\d+))\s*", re.I)
_TRAIL_MULT = re.compile(r"\s*[x×*·]\s*10\s*\^\s*\(?\s*([+-]?\d+)\s*\)?\s*$", re.I)


def _split_multiplier(unit):
    """'×10⁻⁶ cm/s', '(1×10^-6 cm/s)', 'cm/s ×10⁻⁶' -> (1e-6, 'cm/s')."""
    s = _nfkc(unit).translate(_DASHES).strip() if unit is not None else ""
    if len(s) > 2 and s[0] in "([（" and s[-1] in ")]）":
        s = s[1:-1].strip()
    exp, m = None, _LEAD_MULT.match(s)
    if m and re.search(r"[a-zA-Zµμ%]", s[m.end():]):
        exp, s = next(g for g in m.groups() if g is not None), s[m.end():]
    else:
        m = _TRAIL_MULT.search(s)
        if m and re.search(r"[a-zA-Zµμ%]", s[:m.start()]):
            exp, s = m.group(1), s[:m.start()]
    if exp is None:
        return 1.0, s
    return (10.0 ** int(exp) if abs(int(exp)) <= 300 else float("nan")), s


def _unit_dims(u):
    """Canonical unit string -> (dimension signature, factor to base units), or None."""
    parts = u.split("/")
    exps, factor = {}, 1.0
    for sign, toks in ((1, parts[0].split(".") if parts[0] else []), (-1, parts[1:])):
        for tok in toks:
            if tok in _MOLAR_TOKENS:
                dims, f = {"amount": 1, "vol": -1}, _MOLAR_TOKENS[tok]
            elif tok in _ATOMIC_UNITS:
                dims, f = {_ATOMIC_UNITS[tok][0]: 1}, _ATOMIC_UNITS[tok][1]
            else:
                return None
            for dim, e in dims.items():
                exps[dim] = exps.get(dim, 0) + sign * e
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
    leftover = re.sub(r"\^?\s*[-−]\s*1(?![0-9])", "", _nfkc(rest).translate(_DASHES))
    if re.search(r"\d", leftover) or (rest.strip() and not u):
        # a unit this scorer cannot read ('µL/min/10^6 cells', a unit word it does not know):
        # comparable only to the same spelling, never to the empty or another unit
        return "unit:?" + re.sub(r"\s+", "", _nfkc(rest).casefold()), value, False
    if u in _MOLAR_TO_NM:
        return "molar", value * _MOLAR_TO_NM[u], False
    if u in ("%", "percent"):
        return "ratio", value * 0.01, False
    if u == "fraction":
        return "ratio", value, False
    if u in _TIME_TO_H:
        return "time", value * _TIME_TO_H[u], not _is_pow10(_TIME_TO_H[u])
    dims = _unit_dims(u)
    if dims and dims[0] == "amount+1.vol-1":           # nmol/mL, µmol/L ...: molar
        return "molar", value * dims[1] * 1e9, not _is_pow10(dims[1])
    if dims and dims[0] == "mass+1.vol-1" and mw:      # µg/mL with a structure: molar via MW
        return "molar", value * dims[1] / mw * 1e9, True
    if dims:
        return "dim:" + dims[0], value * dims[1], not _is_pow10(dims[1])
    return "unit:" + u, value, False


def _looks_like_unit(text):
    """A unit (optionally with a decimal multiplier), not a quantity such as '72 h' or '10 mg/kg'."""
    t = _nfkc(text).strip()
    if not t or (re.match(r"\d", t) and not _LEAD_MULT.match(t.translate(_DASHES))):
        return False
    mult, rest = _split_multiplier(t)
    u = _norm_unit(rest)
    return bool(u) and (u in _MOLAR_TO_NM or u in _MASS_TO_G_PER_L or u in _TIME_TO_H
                        or u in ("%", "percent", "fraction") or _unit_dims(u) is not None)


# Record context (key v2). A field counts only when the gold fills it ("gold-defined fields"):
# inferring the right species is rewarded, filling in a wrong one is not, leaving it out when
# the gold has it is a miss.
_CONTEXT_FIELDS = ("assay_type", "species", "matrix", "variant", "ph", "dose", "route", "time")
_FIELD_ALIASES = {"cell_line": "matrix", "cell_line_or_matrix": "matrix", "mutant": "variant",
                  "time_point": "time", "incubation_time": "time"}
_SPECIES = {"human": "human", "humans": "human", "homosapiens": "human", "人": "human", "ヒト": "human",
            "rat": "rat", "rats": "rat", "大鼠": "rat", "ラット": "rat",
            "mouse": "mouse", "mice": "mouse", "小鼠": "mouse", "マウス": "mouse",
            "dog": "dog", "dogs": "dog", "beagle": "dog", "beagles": "dog", "beagledog": "dog", "beagledogs": "dog",
            "犬": "dog", "比格犬": "dog", "イヌ": "dog",
            "monkey": "monkey", "monkeys": "monkey", "猴": "monkey", "サル": "monkey",
            "cyno": "cynomolgus", "cynomolgus": "cynomolgus", "cynomolgusmonkey": "cynomolgus", "食蟹猴": "cynomolgus",
            "rhesus": "rhesus", "rhesusmonkey": "rhesus", "恒河猴": "rhesus",
            "minipig": "minipig", "minipigs": "minipig", "rabbit": "rabbit", "rabbits": "rabbit", "兔": "rabbit",
            "hamster": "hamster", "hamsters": "hamster", "guineapig": "guineapig", "guineapigs": "guineapig"}
# Strain and sex qualifiers dropped before the species lookup ('SD rats', 'ICR mice', '雄性SD大鼠').
_STRAINS = ("spraguedawley", "sd", "wistar", "lewis", "icr", "cd1", "cd-1", "balbc", "c57bl6", "c57", "nude",
            "scid", "kunming", "昆明", "male", "female", "雄性", "雌性", "雄", "雌")
_STRAINS_NORM = None   # filled below, after _norm_text is defined
_MATRICES = {"livermicrosomes": "liver_microsomes", "livermicrosome": "liver_microsomes", "microsomes": "liver_microsomes",
             "microsome": "liver_microsomes", "lm": "liver_microsomes", "肝微粒体": "liver_microsomes",
             "肝ミクロソーム": "liver_microsomes", "hepatocytes": "hepatocytes", "hepatocyte": "hepatocytes",
             "肝细胞": "hepatocytes", "肝細胞": "hepatocytes", "plasma": "plasma", "血浆": "plasma", "血漿": "plasma",
             "blood": "blood", "wholeblood": "blood", "全血": "blood", "serum": "serum", "血清": "serum"}
_MATRIX_SPECIES = {"hlm": ("liver_microsomes", "human"), "rlm": ("liver_microsomes", "rat"),
                   "mlm": ("liver_microsomes", "mouse"), "dlm": ("liver_microsomes", "dog"),
                   "cylm": ("liver_microsomes", "cynomolgus"), "mklm": ("liver_microsomes", "monkey"),
                   "hh": ("hepatocytes", "human"), "rh": ("hepatocytes", "rat")}
_ASSAY_TYPES = {"biochemical": "biochemical", "biochem": "biochemical", "enzymatic": "biochemical",
                "enzyme": "biochemical", "enzymeinhibition": "biochemical", "生化": "biochemical", "酶": "biochemical",
                "酶活": "biochemical", "cell": "cell", "cells": "cell", "cellular": "cell", "cellbased": "cell",
                "细胞": "cell", "細胞": "cell", "invivo": "invivo", "体内": "invivo", "binding": "binding", "结合": "binding"}
_ASSAY_SUFFIXES = ("assays", "assay", "level", "水平", "实验", "活性", "法")
_ROUTES = {"iv": "iv", "intravenous": "iv", "intravenously": "iv", "ivbolus": "iv", "intravenousinjection": "iv",
           "静脉": "iv", "静脉注射": "iv", "静注": "iv", "尾静脉注射": "iv", "静脈内": "iv",
           "po": "po", "oral": "po", "orally": "po", "peroral": "po", "oralgavage": "po", "gavage": "po", "ig": "po",
           "口服": "po", "灌胃": "po", "口服灌胃": "po", "経口": "po",
           "ip": "ip", "intraperitoneal": "ip", "腹腔": "ip", "腹腔注射": "ip",
           "sc": "sc", "subcutaneous": "sc", "皮下": "sc", "皮下注射": "sc"}
_ROUTE_SUFFIXES = ("administration", "dosing", "给药", "投与", "投药")
_WILD_TYPE = {"wt", "wildtype", "野生型"}
_STRAINS_NORM = tuple(_norm_text(q) for q in _STRAINS)


@lru_cache(maxsize=50_000)
def _species_of(text):
    t = _norm_text(text)
    for _ in range(3):   # strip stacked qualifiers: '雄性SD大鼠', 'male Wistar rats'
        for q in _STRAINS_NORM:
            if t != q and t.startswith(q):
                t = t[len(q):]
            if t != q and t.endswith(q):
                t = t[:-len(q)]
    return _SPECIES.get(t, t)


@lru_cache(maxsize=50_000)
def _matrix_species(text):
    """'HLM' / 'human liver microsomes' / '人肝微粒体' / 'A549 cells' -> (matrix, species or None)."""
    t = _norm_text(text)
    if t in _MATRIX_SPECIES:
        return _MATRIX_SPECIES[t]
    if t in _MATRICES:
        return _MATRICES[t], None
    for word, species in sorted(_SPECIES.items(), key=lambda kv: -len(kv[0])):
        if t.startswith(word) and t[len(word):] in _MATRICES:
            return _MATRICES[t[len(word):]], species
    for suffix in ("celllines", "cellline", "cells", "cell", "细胞株", "细胞系", "细胞"):
        if t.endswith(suffix) and len(t) > len(suffix):
            return t[:-len(suffix)], None
    return t, None


@lru_cache(maxsize=50_000)
def _assay_type(text):
    t = _norm_text(text)
    for _ in range(2):
        for suffix in _ASSAY_SUFFIXES:
            if t.endswith(suffix) and len(t) > len(suffix):
                t = t[:-len(suffix)]
    return _ASSAY_TYPES.get(t, t)


@lru_cache(maxsize=50_000)
def _known_context(text):
    """Context that a header token unambiguously names: '(HLM)', '(rat)', 'cellular', '(iv)'."""
    t = _norm_text(text)
    if not t:
        return {}
    if _species_of(text) in _SPECIES.values():
        return {"species": _species_of(text)}
    if _assay_type(text) in _ASSAY_TYPES.values():
        return {"assay_type": _assay_type(text)}
    if t in _ROUTES:
        return {"route": _ROUTES[t]}
    if t in _MATRIX_SPECIES or t in _MATRICES or any(
            t.startswith(w) and t[len(w):] in _MATRICES for w in _SPECIES):
        matrix, species = _matrix_species(text)
        return {"matrix": matrix, **({"species": species} if species else {})}
    return {}


def _quantity(text):
    """'1 µM' -> ('conc', ...), '72 h' -> ('time', ...), '10 mg/kg' -> ('dose', ...), else None."""
    m = _QUANTITY.match(_nfkc(text))
    if not m or not _looks_like_unit(m.group(2)):
        return None
    cls, val, _ = _to_canonical(float(m.group(1)), m.group(2))
    if cls in ("molar", "ratio") or cls == "dim:mass+1.vol-1":
        return "conc", f"{cls}:{val:.4g}"
    if cls == "time":
        return "time", f"{val:.4g}"
    if cls.startswith("dim:ratio") or _norm_unit(m.group(2)) in ("mg/kg", "ug/kg", "g/kg", "umol/kg", "mmol/kg"):
        return "dose", f"{cls}:{val:.4g}"
    return None


def _endpoint_parts(raw, target=""):
    e, ctx, unit, condition = _endpoint_parts_cached(str(raw if raw is not None else ""), str(target or ""))
    return e, dict(ctx), unit, condition


@lru_cache(maxsize=100_000)
def _endpoint_parts_cached(raw, target):
    """
    Endpoint header -> (endpoint, context, header_unit, header_condition). Content in brackets
    and next to the name is removed only when it is recognized -- a unit, a statistic ('n=3',
    'mean ± SD'), a test concentration ('1 µM' -> condition), pH, an incubation time or dose,
    or known context (species, matrix, assay type, route) -- so that two columns that differ
    in anything unrecognized (a cell line, a mutant, a strain) never collapse to one key.
    Transport directions and AUC intervals are canonicalized and kept in the name.
    """
    s = _nfkc(raw)
    ctx, unit, condition = {}, "", ""
    s = _DIRECTION.sub(lambda m: " ⟦ab⟧ " if m.group(1) else " ⟦ba⟧ ", s)
    s = _AUC.sub(lambda m: "AUC0" + {"∞": "inf", "infinity": "inf", "last": "t"}.get(
        (m.group(1) or m.group(2) or "").casefold(), (m.group(1) or m.group(2) or "").casefold()) + " ", s)
    ph = _PH.search(s) or _LOGD_PH.search(s)
    if ph:
        ctx["ph"] = f"{float(ph.group(ph.lastindex)):.1f}"
    s = _LOGD_PH.sub(r"\1", _PH.sub("", s))
    at = _AT_CONC.search(s)
    if at:
        q = _quantity(at.group(1) + " " + at.group(2))
        if q and q[0] == "conc":
            condition = q[1]
        s = s[:at.start()] + s[at.end():]
    s = _STAT_INLINE.sub("", s)
    kept = []

    def piece(p, name_so_far):
        nonlocal unit, condition
        p = p.strip()
        if not p or _STATISTIC.match(p):
            return
        if p in ("⟦ab⟧", "⟦ba⟧"):
            kept.append(p)
            return
        if _looks_like_unit(p):
            unit = p   # the last unit-like bracket wins
            return
        q = _quantity(p)
        if q:
            kind, value = q
            if kind == "conc":
                condition = condition or value
            else:
                ctx.setdefault(kind, value)
            return
        if re.fullmatch(r"\d+(?:\.\d+)?", p) and re.search(r"(?i)log\s*d|solubility|溶解度", name_so_far):
            ctx.setdefault("ph", f"{float(p):.1f}")
            return
        known = _known_context(p.strip())
        if known:
            for k, v in known.items():
                ctx.setdefault(k, v)
            return
        if _MUTATION.fullmatch(p.strip()) or (p and all(_MUTATION.fullmatch(x.strip()) for x in re.split(r"[/,;+&]", p))):
            ctx.setdefault("variant", _variant(p, target))
            return
        kept.append(p)   # unrecognized qualifier: stays part of the endpoint

    while True:
        m = _BRACKETS.search(s)
        if not m:
            break
        before = s[:m.start()]
        kept_before = len(kept)
        for p in re.split(r"[,;，；、]", m.group(1)):
            piece(p, before)
        s = before + " " + " ".join(kept[kept_before:]) + " " + s[m.end():]
        del kept[kept_before:]
    # outside brackets: trailing '/nM', ', µM', '值', '%', leading/trailing context words
    m = re.search(r"\s*[/,，]\s*([^/,，]+)$", s)
    if m and _looks_like_unit(m.group(1)):
        unit, s = m.group(1), s[:m.start()]
    s = re.sub(r"值\s*$", "", s.strip())
    words = s.split()
    while len(words) > 1 and _known_context(words[0]):   # cached
        for k, v in _known_context(words.pop(0)).items():
            ctx.setdefault(k, v)
    while len(words) > 1 and _known_context(words[-1]):
        for k, v in _known_context(words.pop()).items():
            ctx.setdefault(k, v)
    e = _norm_text(" ".join(words))
    t = _norm_text(target)
    if t and e != t:
        if e.startswith(t) and e[len(t):]:
            e = e[len(t):]
        elif e.endswith(t) and e[:-len(t)]:
            e = e[:-len(t)]
    if e not in _ENDPOINT_SYNONYMS and e.strip("%") in _ENDPOINT_SYNONYMS:
        e = e.strip("%")
    return _ENDPOINT_SYNONYMS.get(e, e), tuple(ctx.items()), unit, condition


def _norm_endpoint(s):
    return _endpoint_parts(s)[0]


def _variant(v, target=""):
    """'L858R/T790M' == 'T790M, L858R'; 'wild type' == 'WT'; tokens equal to the target are dropped."""
    t = _norm_text(target)
    whole = _norm_text(v)
    if not whole:
        return ""
    if whole in _WILD_TYPE or (t and whole.startswith(t) and whole[len(t):] in _WILD_TYPE):
        return "wt"
    toks = sorted({_norm_text(x) for x in re.split(r"[/,;+&\s]+", _nfkc(v)) if _norm_text(x) and _norm_text(x) != t})
    if toks and set(toks) <= _WILD_TYPE:
        return "wt"
    return "|".join(toks)


def _split_target(raw_target):
    """'EGFR L858R/T790M' -> ('EGFR', 'L858R/T790M'); a target that is only a mutation is kept."""
    s = _nfkc(raw_target)
    muts = _MUTATION.findall(s)
    rest = _MUTATION.sub(" ", s)
    rest = re.sub(r"[\s/,;+&\-()\[\]]+", " ", rest).strip()
    if muts and _norm_text(rest):
        return rest, "/".join(muts)
    return s, ""


def _context(m, from_endpoint, target=""):
    """Normalized context fields of a measurement -> tuple of (field, value) pairs."""
    raw = dict(m)
    for alias, field in _FIELD_ALIASES.items():
        if raw.get(field) in (None, "") and raw.get(alias) not in (None, ""):
            raw[field] = raw[alias]
    ctx = {}
    for field in _CONTEXT_FIELDS:
        v = raw.get(field)
        if v is None or isinstance(v, (dict, list, bool)) or str(v).strip() == "":
            continue
        v = str(v)
        if field == "species":
            ctx[field] = _species_of(v)
        elif field == "matrix":
            matrix, species = _matrix_species(v)
            ctx[field] = matrix
            if species:
                ctx.setdefault("species", species)
        elif field == "assay_type":
            ctx[field] = _assay_type(v)
        elif field == "route":
            t = _norm_text(v)
            for suffix in _ROUTE_SUFFIXES:
                if t.endswith(suffix) and t[:-len(suffix)] in _ROUTES:
                    t = t[:-len(suffix)]
            ctx[field] = _ROUTES.get(t, t)
        elif field == "variant":
            ctx[field] = _variant(v, target)
        elif field == "ph":
            num = re.search(r"\d+(?:\.\d+)?", _nfkc(v))
            ctx[field] = f"{float(num.group(0)):.1f}" if num else _norm_text(v)
        elif field in ("dose", "time"):
            unit_field = raw.get(field + "_unit")
            text = f"{v} {unit_field}" if unit_field not in (None, "") else str(v)
            q = _quantity(text)
            ctx[field] = q[1] if q and q[0] == field else _norm_text(text)
    for field, value in from_endpoint.items():   # context written in the endpoint header
        ctx.setdefault(field, value)
    return tuple(sorted((k, v) for k, v in ctx.items() if v))


_NUM = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?"   # unambiguous: no catastrophic backtracking
_RANGE_SEP = r"(?:-|–|—|~|〜|to|至)"
_RANGE = re.compile(rf"^({_NUM})\s*{_RANGE_SEP}\s*({_NUM})$", re.I)
# a range with the unit after each bound: '10 µM–30 µM'
_RANGE_UNITS = re.compile(rf"^({_NUM})\s*([^\d\s~〜–—\-][^~〜–—]*?)\s*{_RANGE_SEP}\s*({_NUM})\s*(\S.*)$", re.I)
_PREFIX = re.compile(r"^(<=|>=|≤|≥|≦|≧|⩽|⩾|<|>|~|≈|=)\s*")
_WITH_UNIT = re.compile(rf"^({_NUM}(?:\s*{_RANGE_SEP}\s*{_NUM})?)\s*([^\d\s.+-].*)$", re.I)
_TIMES_TEN = re.compile(r"\s*[x×*·]\s*10\s*\^?\s*\(?\s*([+-]?\d+)\s*\)?", re.I)
_E_SPACED = re.compile(r"(\d)\s+[eE]\s*([+-]?\d+)(?![\d.])")
_PM_TIMES_TEN = re.compile(rf"^\(?\s*({_NUM})\s*(?:±|\+/-|\+-)\s*{_NUM}\s*\)?\s*[x×*·]\s*10\s*\^?\s*\(?\s*([+-]?\d+)\s*\)?", re.I)
_SUPERSCRIPTS = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺", "0123456789-+")
_PLUS_MINUS = re.compile(rf"\s*\(?\s*(?:±|\+/-|\+-)\s*{_NUM}\s*\)?|\s*\(\s*n\s*=\s*\d+\s*\)", re.I)
_FOOTNOTE = re.compile(r"\s*(?:[*†‡§¶#]+|[ᵃ-ᶻ]+)$")
_VALUE_WITH_RANGE = re.compile(rf"^({_NUM})\s*\(\s*{_NUM}\s*{_RANGE_SEP}\s*{_NUM}\s*\)$", re.I)


def _header_unit(endpoint):
    """Unit written in the column header / endpoint name, e.g. 'IC50 (µM)' -> 'µM'."""
    return _endpoint_parts(endpoint)[2]


def _half_unit(text):
    """Half a unit in the last printed digit: '7.26' -> 0.005, '50' -> 0.5, '1.2e-5' -> 5e-7."""
    try:
        return 0.5 * 10.0 ** Decimal(text).as_tuple().exponent
    except (InvalidOperation, TypeError, ValueError, OverflowError):
        return 0.0


def _half_unit_safe(text):
    try:
        return _half_unit(text)
    except OverflowError:
        return 0.0


def _decimal_commas(s):
    """'1,200' -> '1200' (thousands); '0,5' and '1,25' -> '0.5', '1.25' (decimal comma)."""
    s = re.sub(r"(?<![\d.,])0,(\d)", r"0.\1", s)
    while re.search(r"\d,\d{3}(?!\d)", s):
        s = re.sub(r"(\d),(\d{3})(?!\d)", r"\1\2", s)
    return re.sub(r"(\d),(\d)", r"\1.\2", s)


def _parse_value(raw, relation):
    """
    -> (numbers, half_units, relation, category, inline_unit). Accepts numbers, [lo, hi], and
    strings such as '1,200', '0,5', '>10', '＜10', '10-30', '10~30', '10 to 30', '10 µM–30 µM',
    '12 ± 3' (mean kept), '12*' and '12ᵃ' (footnote marks), '12 (10–15)', '1.2 × 10⁻⁵',
    '1.2 X 10-8', '1.2 E-8', '1.2×10<sup>-8</sup>', '(1.2 ± 0.3) × 10⁻⁸', '10 µM' (unit written
    in the value); half_units is the printed precision of each number (store gold values as
    printed strings to keep trailing zeros). Anything else non-empty ('NT', 'inactive', a binned
    'A' or '+++') is a categorical value compared as text, together with its relation and unit.
    """
    if isinstance(raw, bool) or raw is None or isinstance(raw, dict):
        return None
    items = list(raw) if isinstance(raw, (list, tuple)) else [raw]
    if not 1 <= len(items) <= 2:
        return None
    if isinstance(raw, (list, tuple)) or isinstance(raw, (int, float)):
        try:
            return (tuple(float(v) for v in items),
                    tuple(_half_unit_safe(repr(v) if isinstance(v, float) else str(v)) for v in items), relation, None, "")
        except (TypeError, ValueError, OverflowError):
            return None
    if len(str(raw)) > MAX_FIELD_CHARS:
        return None
    s = _FOOTNOTE.sub("", re.sub(r"\s+", " ", str(raw)).strip())
    s = _nfkc(s).strip()
    s = re.sub(r"^[‐‑‒–—−﹣－]\s*(?=[\d.])", "-", s).replace("−", "-")
    s = _decimal_commas(s)
    pm = _PM_TIMES_TEN.match(s)
    if pm:   # '(1.2 ± 0.3) × 10⁻⁸': the exponent applies to the mean
        s = f"{pm.group(1)}e{pm.group(2)}" + s[pm.end():]
    s = _TIMES_TEN.sub(lambda mm: "e" + mm.group(1), s)
    s = _E_SPACED.sub(r"\1e\2", s)
    s = _PLUS_MINUS.sub("", s).strip()
    s = _FOOTNOTE.sub("", s)
    vr = _VALUE_WITH_RANGE.match(s)
    if vr:   # '12 (10–15)': the value, with its range in brackets
        s = vr.group(1)
    m = _PREFIX.match(s)
    if m:
        relation, s = (m.group(1) if relation in ("", "=") else relation), s[m.end():]
    unit = ""
    ru = _RANGE_UNITS.match(s)
    if ru and _norm_unit(ru.group(2)) == _norm_unit(ru.group(4)):
        s, unit = f"{ru.group(1)}-{ru.group(3)}", ru.group(4).strip()
    wu = None if unit or _RANGE.match(s) or re.fullmatch(_NUM, s, re.I) else _WITH_UNIT.match(s)
    if wu and re.search(r"[a-zA-Zµμ%]", wu.group(2)):
        s, unit = wu.group(1).strip(), wu.group(2).strip()
    r = _RANGE.match(s)
    if r:
        return ((float(r.group(1)), float(r.group(2))), (_half_unit_safe(r.group(1)), _half_unit_safe(r.group(2))),
                relation, None, unit)
    try:
        return (float(s),), (_half_unit_safe(s),), relation, None, unit
    except ValueError:
        text = re.sub(r"(?<!\d)\.|\.(?!\d)", "", re.sub(r"[^0-9a-z.+\-<>%]+", "", s.casefold()))
        if not text and not unit:
            return None
        token = text + ("|" + _norm_unit(unit) if unit else "")
        return (), (), relation, token, ""


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
    the record-key-v2 fields (assay_type, species, matrix, variant, ph, dose, route, time), also
    when written in the header ('CLint (HLM)', 'logD7.4'); they count only where the gold has them.
    """
    if any(len(str(m.get(f) or "")) > MAX_FIELD_CHARS for f in ("unit", "endpoint", "target", "relation", "conc", "conc_unit")):
        return None
    parsed = _parse_value(m.get("value"), _nfkc(m.get("relation") or "").strip())
    if parsed is None:
        return None
    values, halves, relation, category, inline_unit = parsed
    if len(values) > 2 or not all(math.isfinite(v) for v in values):
        return None
    raw_target, target_variant = _split_target(m.get("target")) if MATCH_TARGET else ("", "")
    endpoint, header_ctx, header_unit, header_condition = _endpoint_parts(m.get("endpoint"), raw_target)
    target = _norm_text(raw_target) if MATCH_TARGET else ""
    if (str(m.get("endpoint") or "").strip() and not endpoint) or \
            (MATCH_TARGET and str(m.get("target") or "").strip() and not target):
        return None   # e.g. a label in a script this scorer does not keep: fail closed
    if target_variant and not str(m.get("variant") or "").strip():
        header_ctx.setdefault("variant", _variant(target_variant, raw_target))
    unit = m.get("unit")
    if not str(unit or "").strip():   # unit written in the value, else in the column header
        unit = inline_unit or header_unit or ("%" if endpoint in _PERCENT_ENDPOINTS else
                                               "fraction" if endpoint in _FRACTION_ENDPOINTS else "")
    bounds = tuple((v - h, v + h) for v, h in zip(values, halves))
    relation = "range" if len(values) == 2 else _RELATIONS.get(relation, "?" + relation)
    converted = False
    if category is not None:
        relation, unit_class = "cat:" + (relation if relation not in ("", "=") else "") + category, ""
    elif endpoint in _P_ENDPOINTS and _norm_unit(unit) in _PX_UNITS:
        if not all(-30 < v < 30 for v in values):
            return None   # not a plausible pX; also keeps 10 ** (9 - v) finite
        endpoint, unit_class, converted = _P_ENDPOINTS[endpoint], "molar", True
        values, relation = tuple(10.0 ** (9.0 - v) for v in values), _FLIP.get(relation, relation)
        bounds = tuple((10.0 ** (9.0 - hi), 10.0 ** (9.0 - lo)) for lo, hi in bounds)
    else:
        canon = [_to_canonical(v, unit, mw) for v in values]
        unit_class, converted = canon[0][0], canon[0][2]
        scale = [c[1] / v if v else _to_canonical(1.0, unit, mw)[1] for c, v in zip(canon, values)]
        values = tuple(c[1] for c in canon)
        bounds = tuple((lo * k, hi * k) for (lo, hi), k in zip(bounds, scale))
    if not all(math.isfinite(v) for v in values):
        return None   # e.g. '1e400', or a ×10^400 multiplier
    order = sorted(range(len(values)), key=lambda i: values[i])
    values, bounds = tuple(values[i] for i in order), tuple(bounds[i] for i in order)
    conc, conc_unit = m.get("conc"), m.get("conc_unit")
    condition = header_condition
    if conc not in (None, ""):
        try:
            cls, cval, _ = _to_canonical(float(conc), conc_unit)
            condition = f"{cls}:{cval:.4g}"
        except (TypeError, ValueError, OverflowError):
            q = _quantity(f"{conc} {conc_unit or ''}") if isinstance(conc, (str, int, float)) else None
            condition = q[1] if q and q[0] == "conc" else "?:" + _norm_text(f"{conc}{conc_unit or ''}")
    return endpoint, target, relation, unit_class, condition, values, converted, bounds, _context(m, header_ctx, raw_target)

# ---------------------------------------------------------------------------
# Per-document scoring
# ---------------------------------------------------------------------------

_LABEL_LIST = re.compile(r"[;；、]")
_LABEL_PREFIX = re.compile(r"^(?:compound|cpd|example|ex|化合物|实施例|實施例|実施例)(?:\.\s*|\s+|(?=\d))")


def _norm_label(s):
    """
    'Compound 12a' / 'cpd. 12a' / 'Ex. 12a' / '12 a' -> '12a'. A separator between two digits
    is kept ('1-11' != '11-1' != '111'); a word that merely starts with 'ex' is not a prefix.
    A list of labels ('1;2', '12a、12b') is not a label: it links to nothing.
    """
    if not isinstance(s, (str, int, float)) or isinstance(s, bool):
        return ""
    if isinstance(s, str) and _LABEL_LIST.search(s.strip().strip(";；、 ")):
        return ""
    s = _LABEL_PREFIX.sub("", _nfkc(s).casefold().strip()).translate({0x2032: "'", 0x2033: "''", 0x2019: "'"})
    s = re.sub(r"(?<=\d)[^0-9a-z'\u3400-\u9fff]+(?=\d)", "§", s)
    return re.sub(r"[^0-9a-z'\u3400-\u9fff§]+", "", s).replace("§", "-")


def _capped(items, cap):
    """
    Items beyond the cap are kept as empty placeholders: they count, but are never valid.
    Malformed items (not an object) count the same way; a non-list counts as nothing.
    """
    items = list(items) if isinstance(items, list) else []
    items = [x if isinstance(x, dict) else {} for x in items]
    return items[:cap] + [{}] * max(0, len(items) - cap)


def _compound_table(compounds):
    """compound id -> smiles."""
    return {_norm_label(c["id"]): c.get("smiles") for c in compounds if c.get("id") is not None and _norm_label(c["id"])}


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
        try:
            norm = normalize_measurement(m, mol_weight(table[cid]) if cid in table else None) if isinstance(m, dict) else None
        except (ArithmeticError, ValueError, TypeError, re.error):
            norm = None
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


MATCH_EXACT_PAIRS = 250_000   # gold x pred pairs per record key; larger groups use the interval greedy


def _window(gold):
    """Interval of predicted (first) values that can be close to this gold value, either rule."""
    (gv, _, gb, _) = gold
    g, (lo, hi) = gv[0], gb[0]
    tol = abs(g) * VALUE_REL_TOL / (1 - VALUE_REL_TOL) + 1e-12
    ends = (lo * (1 - 1e-9), lo * (1 + 1e-9), hi * (1 - 1e-9), hi * (1 + 1e-9), g - tol, g + tol)
    return min(ends), max(ends)


def _kuhn(adj, n_pred):
    """Maximum bipartite matching by augmenting paths (iterative, no recursion limit)."""
    owner = [-1] * n_pred
    matched = 0
    for start in range(len(adj)):
        seen = set()
        stack = [(start, iter(adj[start]))]
        path = []
        found = False
        while stack and not found:
            g, it = stack[-1]
            for j in it:
                if j in seen:
                    continue
                seen.add(j)
                if owner[j] < 0:
                    path.append((g, j))
                    found = True
                    break
                path.append((g, j))
                stack.append((owner[j], iter(adj[owner[j]])))
                break
            else:
                stack.pop()
                if path:
                    path.pop()
        if found:
            for g, j in path:   # flip the augmenting path
                owner[j] = g
            matched += 1
    return matched


def _max_matching(gold_vals, pred_vals):
    """
    One-to-one matching between gold and predicted values of one record key. Exact (maximum
    bipartite matching over window-pruned candidates) for ordinary groups; for very large
    groups an interval greedy (golds by right end of their window, each takes the first free
    close prediction), which can only undercount, so a flood of records cannot slow scoring
    down or raise the score.
    """
    if not gold_vals or not pred_vals:
        return 0
    num = sorted((p[0][0], j) for j, p in enumerate(pred_vals) if p[0])
    keys = [x for x, _ in num]
    cats = [j for j, p in enumerate(pred_vals) if not p[0]]

    def candidates(g):
        if not g[0]:
            return cats
        lo, hi = _window(g)
        return [num[i][1] for i in range(bisect_left(keys, lo), bisect_right(keys, hi))]

    if len(gold_vals) * len(pred_vals) <= MATCH_EXACT_PAIRS:
        adj = [[j for j in candidates(g) if _close(g, pred_vals[j])] for g in gold_vals]
        return _kuhn(adj, len(pred_vals))
    # interval greedy over the sorted numeric predictions (categoricals: one shared "line");
    # nxt[] is a union-find "next untaken index" so taken predictions are skipped in O(α)
    lines = {False: [j for _, j in num], True: cats}
    nxt = {k: list(range(len(v) + 1)) for k, v in lines.items()}

    def free_from(k, i):
        root = i
        while nxt[k][root] != root:
            root = nxt[k][root]
        while nxt[k][i] != root:
            nxt[k][i], i = root, nxt[k][i]
        return root

    order = sorted(range(len(gold_vals)), key=lambda i: _window(gold_vals[i])[1] if gold_vals[i][0] else math.inf)
    matched = 0
    for i in order:
        g = gold_vals[i]
        is_cat = not g[0]
        line = lines[is_cat]
        if is_cat:
            a, b = 0, len(line)
        else:
            lo, hi = _window(g)
            a, b = bisect_left(keys, lo), bisect_right(keys, hi)
        k, tried = free_from(is_cat, a), 0
        while k < b and tried < 64:
            if _close(g, pred_vals[line[k]]):
                nxt[is_cat][k] = k + 1
                matched += 1
                break
            tried += 1
            k = free_from(is_cat, k + 1)
    return matched


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
    gold = gold if isinstance(gold, dict) else {}
    pred = pred if isinstance(pred, dict) else {}
    clean = lambda c: dict(c, smiles=c["smiles"] if isinstance(c.get("smiles"), str) else None)
    g_compounds = [clean(c) for c in (gold.get("compounds") or []) if isinstance(c, dict)]
    g_measurements = [m for m in (gold.get("measurements") or []) if isinstance(m, dict)]
    g_table = _compound_table(g_compounds)
    g_smiles = [c.get("smiles") for c in g_compounds]
    g_keys = [structure_keys(s) for s in g_smiles]
    g_valid = [k for k in g_keys if k[0] is not None]
    n_gold_unscorable = len(g_keys) - len(g_valid)
    # Predictions about a gold compound this scorer cannot represent (a drawing SMILES cannot
    # express) are set aside, tied by compound label: the gold side is excluded from recall,
    # so they are excluded from precision. Unlabelled extras are never forgiven, so emitting
    # several hedged answers does not pay.
    unscorable = {_norm_label(c.get("id")) for c, k in zip(g_compounds, g_keys) if k[0] is None} - {""}
    p_compounds = [clean(c) for c in _capped(pred.get("compounds"), MAX_PRED_COMPOUNDS)
                   if _norm_label(c.get("id")) not in unscorable]
    p_measurements = [m for m in _capped(pred.get("measurements"), MAX_PRED_RECORDS)
                      if _norm_label(m.get("compound_id")) not in unscorable]
    p_table = _compound_table(p_compounds)
    p_smiles = [c.get("smiles") for c in p_compounds]
    p_keys = [structure_keys(s) for s in p_smiles]
    p_valid = [k for k in p_keys if k[0] is not None]
    struct_tp = _multiset_tp([k for k, _ in g_valid], [k for k, _ in p_valid])
    n_pred_struct = len(p_keys)
    if g_keys and not g_valid:
        n_pred_struct = 0   # no scorable structure in this document
    elif not g_measurements:
        # Structure-only documents (OCSR crops): a missing answer costs what a wrong one costs.
        # Otherwise F1 = 2c/(c+w+n) rewards abstaining over guessing.
        n_pred_struct = max(n_pred_struct, len(g_valid))
    tanimoto_sum = sum(max((tanimoto(gs, ps) for ps in p_smiles), default=0.0)
                       for gs, gk in zip(g_smiles, g_keys) if gk[0] is not None)
    g_ids = [(_norm_label(c["id"]), k[0]) for c, k in zip(g_compounds, g_keys) if c.get("id") is not None and k[0]]
    p_ids = [(_norm_label(c["id"]), k[0]) for c, k in zip(p_compounds, p_keys) if c.get("id") is not None]
    g_recs = _records(g_measurements, g_table, by_label=False)
    p_recs = _records(p_measurements, p_table, by_label=False)
    record_tp = _record_tp(g_recs, p_recs)
    n_gold_rec_unscorable = sum(1 for k, _ in g_recs if k is None)
    n_pred_rec = len(p_recs)
    g_recs_label = _records(g_measurements, g_table, by_label=True)
    p_recs_label = _records(p_measurements, p_table, by_label=True)
    return {
        "doc_id": gold.get("doc_id"),
        "family_id": gold.get("family_id") or gold.get("doc_id"),
        "slice": gold.get("slice") or "all",
        "doc_type": gold.get("doc_type") or "",
        "tags": sorted(set(gold.get("tags") or [])),
        **{f: str(gold[f]) for f in _GROUP_FIELDS if isinstance(gold.get(f), (str, int)) and gold.get(f) != ""},
        "n_gold_unscorable": n_gold_unscorable,
        "struct_tp": struct_tp,
        "struct_tp_skeleton": _multiset_tp([s for _, s in g_valid], [s for _, s in p_valid]),
        "n_gold_struct": len(g_valid),
        "n_pred_struct": n_pred_struct,
        "n_pred_struct_raw": len(p_keys),
        "n_invalid_struct": len(p_keys) - len(p_valid),
        "n_multifrag_struct": sum(1 for s in p_smiles if is_multifragment(s)),
        "tanimoto_sum": tanimoto_sum,
        "id_link_tp": _multiset_tp(g_ids, p_ids),
        "n_gold_id": len(g_ids),
        "n_pred_id": len(p_ids),
        "record_tp": record_tp,
        "record_tp_label": _record_tp(g_recs_label, p_recs_label),
        "n_gold_rec_label": sum(1 for k, _ in g_recs_label if k is not None),
        "n_pred_rec_label": len(p_recs_label),
        "n_gold_rec": sum(1 for k, _ in g_recs if k is not None),
        "n_gold_rec_unscorable": n_gold_rec_unscorable,
        "n_pred_rec": n_pred_rec,
        "n_orphan_rec": sum(1 for m in p_measurements if _norm_label(m.get("compound_id")) not in p_table),
    }

# ---------------------------------------------------------------------------
# Aggregation, composite score, bootstrap
# ---------------------------------------------------------------------------

# Optional gold fields copied into the per-document rows so that --group-field can use them.
_GROUP_FIELDS = ("office", "language")
_ID_FIELDS = ("doc_id", "family_id", "slice", "doc_type", "tags", "gold") + _GROUP_FIELDS
SCALARS = ("auto", "composite", "struct_f1", "record_f1")
# Guardrail rates reported with paired deltas and bootstrap SEs by `compare` (numerator, denominator).
GUARDRAILS = {"invalid_smiles_rate": ("n_invalid_struct", "n_pred_struct_raw"),
              "multifragment_rate": ("n_multifrag_struct", "n_pred_struct_raw"),
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
        "invalid_smiles_rate": _ratio(get("n_invalid_struct"), get("n_pred_struct_raw")),
        "multifragment_rate": _ratio(get("n_multifrag_struct"), get("n_pred_struct_raw")),
        "id_link_f1": _f1(get("id_link_tp"), get("n_pred_id"), get("n_gold_id"))[0],
        "record_f1": rec_f1,
        "record_precision": rec_p,
        "record_recall": rec_r,
        "record_f1_by_label": _f1(get("record_tp_label"), get("n_pred_rec_label"), get("n_gold_rec_label"))[0],
        "orphan_record_rate": _ratio(get("n_orphan_rec"), get("n_pred_rec")),
    }
    out = {k: float(v) for k, v in out.items()}
    out.update({"scalar": mode + ("/slice-weighted" if slice_weights else ""), "n_docs": len(per_doc),
                "n_gold_struct": get("n_gold_struct"), "n_gold_unscorable": get("n_gold_unscorable"),
                "n_gold_rec": get("n_gold_rec"), "n_gold_rec_unscorable": get("n_gold_rec_unscorable")})
    return out

# Bootstrap. Totals are kept per (slice, family) cell so that slice-weighted scores can be
# computed in every replicate, but whole families (patent family / article, e.g. a paper and
# its SI) are resampled together, stratified by the set of slices a family spans, so every
# replicate keeps the slice composition. With several seed runs per side, seeds are resampled
# too (hierarchical), so seed-to-seed variance enters the interval.


def _count_fields(runs):
    return sorted({k for run in runs for d in run for k in d if k not in _ID_FIELDS})


def _cell(d):
    return (str(d.get("slice") or "all"), str(d.get("family_id") or d["doc_id"]))


def _cluster_matrix(per_doc, order=None, fields=None):
    """-> (cell keys, {field: per-cell totals}) for one run."""
    fields = fields or _count_fields([per_doc])
    keys = [_cell(d) for d in per_doc]
    order = order or sorted(set(keys))
    index = {c: i for i, c in enumerate(order)}
    totals = np.zeros((len(order), len(fields)))
    np.add.at(totals, np.array([index[k] for k in keys]),
              np.array([[d.get(f, 0) for f in fields] for d in per_doc], dtype=float))
    return order, {f: totals[:, j] for j, f in enumerate(fields)}


def _stack_runs(runs, order, fields=None):
    """Seed runs -> {field: (seeds x cells) array}."""
    fields = fields or _count_fields(runs)
    per_run = [_cluster_matrix(r, order, fields)[1] for r in runs]
    return {f: np.stack([t[f] for t in per_run]) for f in per_run[0]}


def _stratified_weights(order, rng):
    """
    (replicates x cells) resampling counts: each family is drawn as a whole (all its cells get
    the family's count), multinomially within strata of families spanning the same slices.
    """
    cells = {}
    for i, (_, fam) in enumerate(order):
        cells.setdefault(fam, []).append(i)
    strata = {}
    for fam, idx in cells.items():
        strata.setdefault(tuple(sorted({order[i][0] for i in idx})), []).append(fam)
    w = np.zeros((BOOTSTRAP_SAMPLES, len(order)))
    for key in sorted(strata):
        fams = sorted(strata[key])
        counts = rng.multinomial(len(fams), np.full(len(fams), 1.0 / len(fams)), size=BOOTSTRAP_SAMPLES)
        for j, fam in enumerate(fams):
            w[:, cells[fam]] = counts[:, [j]]
    return w


def _n_families(order):
    return len({fam for _, fam in order})


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
    """Percentile CI of the primary scalar, resampling whole families within slice strata."""
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
    return _ratio(sum(t.get(num, 0) for t in by_slice.values()), sum(t.get(den, 0) for t in by_slice.values()))


def regroup(per_doc, field="slice"):
    """Use another gold field (e.g. doc_type for the S2 macro-average) as the grouping `slice`."""
    if field == "slice":
        return per_doc
    if field not in ("doc_type",) + _GROUP_FIELDS:
        raise ValueError(f"--group-field {field}: not a groupable field "
                         f"(use slice, doc_type, {', '.join(_GROUP_FIELDS)}, or put it in tags)")
    if per_doc and not any(d.get(field) for d in per_doc):
        raise ValueError(f"--group-field {field}: no gold document has this field")
    return [dict(d, slice=str(d.get(field) or "all")) for d in per_doc]


def check_slice_weights(per_doc, slice_weights, allow_unweighted=False):
    """
    -> (effective weights, weighted slices absent from the data). A slice present in the data
    without a weight is an error unless allow_unweighted: a misspelt slice name would otherwise
    silently drop out of the primary scalar.
    """
    present = {d.get("slice") or "all" for d in per_doc}
    unweighted = sorted(present - set(slice_weights))
    if unweighted and not allow_unweighted:
        raise ValueError(f"slices without a weight: {unweighted} (weights given for {sorted(slice_weights)})")
    used = {s: w for s, w in slice_weights.items() if s in present}
    if not used:
        raise ValueError(f"none of the weighted slices {sorted(slice_weights)} is in the data")
    total = sum(used.values())
    return {s: w / total for s, w in used.items()}, sorted(set(slice_weights) - present)


def paired_compare(base, new, alpha=0.10, scalar="auto", slice_weights=None, require_same_docs=True):
    """
    delta = score(new) - score(base), paired on doc_ids. `base` and `new` are one run or a list
    of seed runs each, scored against the same gold (checked through the per-doc `gold` hash
    and the doc_id sets unless require_same_docs=False, which pairs on the shared doc_ids).
    Every bootstrap replicate resamples whole families within slice strata (shared by both
    sides, which is the pairing) and, separately for each side, its seeds. With one seed on
    one side only (the 1-vs-3 screen) that side's seed variance is borrowed from the other
    side's between-seed spread; with one seed per side, seed variance cannot enter at all and
    the result is only a rough screen. Also returns paired
    deltas with bootstrap SEs for every slice scalar and guardrail rate, and for every
    difficulty tag (gold `tags`, several per document), so that vetoes can be scaled to their
    own noise. Guardrail rates are lower-is-better: veto if delta > +k * se; slices and tags
    are higher-is-better: veto if delta < -k * se.
    """
    base_runs, new_runs = _as_runs(base), _as_runs(new)
    gold_hashes = {d.get("gold") for run in base_runs + new_runs for d in run} - {None}
    doc_sets = {frozenset(d["doc_id"] for d in run) for run in base_runs + new_runs}
    if require_same_docs and (len(gold_hashes) > 1 or len(doc_sets) > 1):
        raise ValueError(f"runs were scored against different gold ({len(gold_hashes)} gold hashes, "
                         f"{len(doc_sets)} distinct doc_id sets); pass require_same_docs=False to pair on shared docs")
    gold_keys = {d["doc_id"]: (d.get("slice") or "all", d.get("family_id") or d["doc_id"]) for d in base_runs[0]}
    ids = set(gold_keys)
    for run in base_runs + new_runs:
        ids &= {d["doc_id"] for d in run}
    assert ids, "no shared doc_ids between the runs"
    keep = lambda run: [dict(d, slice=gold_keys[d["doc_id"]][0], family_id=gold_keys[d["doc_id"]][1])
                        for d in run if d["doc_id"] in ids]
    base_runs, new_runs = [keep(r) for r in base_runs], [keep(r) for r in new_runs]
    mode = resolve_scalar(base_runs[0], scalar)
    fields = _count_fields(base_runs + new_runs)
    order, _ = _cluster_matrix(base_runs[0], fields=fields)
    base_m, new_m = _stack_runs(base_runs, order, fields), _stack_runs(new_runs, order, fields)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    w = _stratified_weights(order, rng)
    base_k, new_k = _seed_weights(len(base_runs), rng), _seed_weights(len(new_runs), rng)
    base_rep = _slice_totals(base_m, order, w, base_k)
    new_rep = _slice_totals(new_m, order, w, new_k)
    base_pt, new_pt = _slice_totals(base_m, order), _slice_totals(new_m, order)
    # A one-seed side (the 1-vs-3 screen) carries no seed variance of its own: borrow it from the
    # other side's seeds, one normal draw per replicate shared by every quantity. Only the
    # global part is borrowed: the spread of the stored seeds' scores also contains item-level
    # run-to-run noise, which the paired document resampling already counts.
    nb, nn = len(base_runs), len(new_runs)
    multi = new_m if nb == 1 < nn else base_m if nn == 1 < nb else None
    z = rng.standard_normal(BOOTSTRAP_SAMPLES) if multi is not None else None
    one_seed = np.ones((BOOTSTRAP_SAMPLES, 1))

    def jitter(f, mats=None):
        """Seed noise for the one-seed side of quantity f (0 when both sides have seeds)."""
        if z is None:
            return 0.0
        mats = multi if mats is None else mats
        seeds = [{k: m[[i]] for k, m in mats.items()} for i in range(next(iter(mats.values())).shape[0])]
        vals = [float(f(_slice_totals(m, order))) for m in seeds]
        reps = [f(_slice_totals(m, order, w, one_seed)) for m in seeds]
        within = np.mean([np.var(reps[i] - reps[j], ddof=1) / 2
                          for i in range(len(seeds)) for j in range(i + 1, len(seeds))])
        var = float(np.var(vals, ddof=1)) - float(within)
        return z * math.sqrt(var) if math.isfinite(var) and var > 0 else 0.0

    primary = lambda t: _primary(t, mode, slice_weights)
    deltas = primary(new_rep) - primary(base_rep) + jitter(primary)
    paired = lambda f: (float(f(new_pt) - f(base_pt)), float(np.std(f(new_rep) - f(base_rep) + jitter(f), ddof=1)))
    guards = {g: paired(lambda t, num=num, den=den: _guard_value(t, num, den)) for g, (num, den) in GUARDRAILS.items()}
    slices = {s: paired(lambda t, s=s: _scalar_from(t[s], mode)) for s in base_pt}
    tags = {}
    tag_docs = {}  # tag -> doc_ids, from the first base run so every run is cut the same way
    for d in base_runs[0]:
        for t in d.get("tags") or []:
            tag_docs.setdefault(t, set()).add(d["doc_id"])
    for tag, members in sorted(tag_docs.items()):
        sub = lambda runs: _stack_runs([[d for d in r if d["doc_id"] in members] for r in runs], order, fields)
        tb, tn = sub(base_runs), sub(new_runs)
        whole = lambda tot: _scalar_from({f: sum(t[f] for t in tot.values()) for f in next(iter(tot.values()))}, mode)
        pt = whole(_slice_totals(tn, order)) - whole(_slice_totals(tb, order))
        rep_d = whole(_slice_totals(tn, order, w, new_k)) - whole(_slice_totals(tb, order, w, base_k))
        if z is not None:
            rep_d = rep_d + jitter(whole, tn if nn > 1 else tb)
        tags[tag] = (float(pt), float(np.std(rep_d, ddof=1)))
    seed_scores = lambda runs: [round(aggregate(r, mode, slice_weights)["score"], 6) for r in runs]
    return {"delta": float(_primary(new_pt, mode, slice_weights) - _primary(base_pt, mode, slice_weights)),
            "ci_lo": float(np.quantile(deltas, alpha / 2)), "ci_hi": float(np.quantile(deltas, 1 - alpha / 2)),
            "p_not_better": float(np.mean(deltas <= 0)), "n_shared": len(ids), "n_clusters": _n_families(order),
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
    with open(path, "rb") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line.decode("utf-8"))   # UnicodeDecodeError is a ValueError
                if not isinstance(row, dict) or not isinstance(row.get("doc_id"), (str, int)) or isinstance(row.get("doc_id"), bool):
                    raise ValueError("not a JSON object with a string or integer doc_id")
                rows.append(row)
            except (ValueError, RecursionError):
                if strict:
                    raise
                bad += 1
    return rows, bad


def _check_pred_schema(pred_rows):
    """
    A prediction file whose records cannot reach any compound is a format error, not a score:
    without this, a converter that writes `compound` instead of `compound_id` scores record F1 = 0
    with no warning.
    """
    ms = [m for d in pred_rows if isinstance(d.get("measurements"), list)
          for m in d["measurements"] if isinstance(m, dict)]
    if ms and not any("compound_id" in m for m in ms):
        hint = " (found key 'compound': convert records_yaml_v1 with yaml_to_jsonl.py)" if any("compound" in m for m in ms) else ""
        raise ValueError(f"schema mismatch: none of the {len(ms)} predicted measurements has compound_id{hint}")
    cs = [c for d in pred_rows if isinstance(d.get("compounds"), list) for c in d["compounds"] if isinstance(c, dict)]
    if cs and not any("smiles" in c for c in cs):
        raise ValueError(f"schema mismatch: none of the {len(cs)} predicted compounds has smiles")
    if cs and not any("id" in c for c in cs) and any("compound_id" in c for c in cs):
        raise ValueError(f"schema mismatch: none of the {len(cs)} predicted compounds has id "
                         "(found 'compound_id': compounds use 'id', measurements use 'compound_id')")
    if not ms and any(k in d for d in pred_rows for k in ("records", "rgroups", "scaffold")):
        raise ValueError("schema mismatch: records_yaml_v1 keys (records/rgroups/scaffold) but no measurements; "
                         "convert with yaml_to_jsonl.py")


def score_files(gold_path, pred_path):
    """Every gold document is scored; a missing or unparseable prediction counts as empty."""
    gold, _ = _read_jsonl(gold_path)
    pred_rows, n_bad = _read_jsonl(pred_path, strict=False)
    pred = {}
    for d in pred_rows:
        pred.setdefault(d.get("doc_id"), d)  # first prediction wins: no best-of-N
    _check_pred_schema(pred_rows)
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


def _line(key, value):
    print(f"{key + ':':24s}{value}")


def main():
    parser = argparse.ArgumentParser(description="Fixed scorer for structure + property extraction")
    sub = parser.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score")
    s.add_argument("--gold", required=True)
    s.add_argument("--pred", required=True)
    s.add_argument("--per-doc-out", default=None)
    c = sub.add_parser("compare")
    c.add_argument("--base", required=True, help="per-doc file(s), comma-separated seeds")
    c.add_argument("--new", required=True, help="per-doc file(s), comma-separated seeds")
    c.add_argument("--alpha", type=float, default=0.10, help="two-sided; 0.05 gives a 95%% CI")
    c.add_argument("--allow-subset", action="store_true",
                   help="pair on shared doc_ids even if the runs were scored against different gold")
    for p in (s, c):
        p.add_argument("--scalar", choices=SCALARS, default="auto")
        p.add_argument("--slice-weights", default=None, help="e.g. inhouse=0.3,biovista=0.3,uspto=0.2,mrbw=0.2")
        p.add_argument("--group-field", default="slice",
                       help="gold field used as the slice for weighting/stratification, e.g. doc_type for S2")
        p.add_argument("--allow-unweighted-slices", action="store_true",
                       help="with --slice-weights, ignore slices that have no weight instead of failing")
    args = parser.parse_args()
    _check_rdkit()
    weights = parse_slice_weights(args.slice_weights)

    if args.cmd == "score":
        per_doc, n_missing = score_files(args.gold, args.pred)
        gold_hash = _sha256(args.gold)
        per_doc = [dict(d, gold=gold_hash) for d in per_doc]
        if args.per_doc_out:
            with open(args.per_doc_out, "w") as f:
                for d in per_doc:
                    f.write(json.dumps(d) + "\n")
        per_doc = regroup(per_doc, args.group_field)
        if weights:
            effective, absent = check_slice_weights(per_doc, weights, args.allow_unweighted_slices)
        summary = aggregate(per_doc, args.scalar, weights)
        lo, hi = bootstrap_ci(per_doc, scalar=args.scalar, slice_weights=weights)
        print("---")
        for k, v in summary.items():
            _line(k, f"{v:.6f}" if isinstance(v, float) else v)
        _line("score_ci95", f"[{lo:.6f}, {hi:.6f}]")
        if weights:
            _line("slice_weights_used", ",".join(f"{k}={v:.3f}" for k, v in sorted(effective.items())))
            if absent:
                _line("slice_weights_absent", ",".join(absent))
        _line("missing_pred_docs", n_missing)
        _line("rdkit_version", RDKIT_VERSION)
        _line("gold_sha256", gold_hash)
        _line("scorer_version", SCORER_VERSION)
        _line("scorer_sha256", _sha256(os.path.abspath(__file__)))
        mode = resolve_scalar(per_doc, args.scalar)
        slices = sorted({d["slice"] for d in per_doc})
        groups = [("slice", n, [d for d in per_doc if d["slice"] == n]) for n in slices] if len(slices) > 1 else []
        groups += [("tag", t, [d for d in per_doc if t in d["tags"]]) for t in sorted({t for d in per_doc for t in d["tags"]})]
        for kind, name, docs in groups:
            sl = aggregate(docs, mode)
            _line(kind + "/" + name, f"score={sl['score']:.6f} n_docs={sl['n_docs']}")
    else:
        load = lambda spec: [regroup(_read_jsonl(p)[0], args.group_field) for p in spec.split(",")]
        base, new = load(args.base), load(args.new)
        if weights:
            check_slice_weights(base[0], weights, args.allow_unweighted_slices)
        r = paired_compare(base, new, alpha=args.alpha, scalar=args.scalar, slice_weights=weights,
                           require_same_docs=not args.allow_subset)
        print("---")
        _line("delta", f"{r['delta']:+.6f}")
        _line("delta_ci%d" % round(100 * (1 - args.alpha)), f"[{r['ci_lo']:+.6f}, {r['ci_hi']:+.6f}]")
        _line("p_not_better", f"{r['p_not_better']:.4f}")
        _line("n_shared_docs", r["n_shared"])
        _line("n_clusters", r["n_clusters"])
        _line("scalar", r["scalar"])
        _line("base_seed_scores", r["base_seed_scores"])
        _line("new_seed_scores", r["new_seed_scores"])
        for group in ("guardrails", "slices", "tags"):
            for name, (d, se) in r[group].items():
                _line(group[:-1] + "/" + name, f"delta={d:+.6f} se={se:.6f}")


if __name__ == "__main__":
    try:
        main()
    except ValueError as e:   # configuration errors (slice weights, mismatched gold): no traceback
        sys.exit(f"error: {e}")
