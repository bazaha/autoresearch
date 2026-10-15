"""
records_yaml_v1 (the table-reading format, docs/deliverable_schema.md section 4) -> the JSONL
that score.py reads, one line per doc_id. One YAML document per table; a file may hold several
(separated by ---), and tables of the same doc_id are merged.

  table:     {doc_id: "...", page: "5", table_id: "T2"}
  scaffold:  {cxsmiles: "O=C(Nc1ccc([*:1])cc1)c1ccncc1"}      # or smiles; null when unreadable
  rgroups:   [{compound_id: "12a", R1: "[H]"}, {compound_id: "12b", R1: "F"}, ...]
  compounds: [{compound_id: "7", smiles: "..."}]               # optional: structures given whole
  legend:    {"A": "<10 nM"}
  records:   [{compound_id: "12a", endpoint: "IC50", target: "EGFR", value: "12.3", unit: "nM"}, ...]

R positions in the scaffold are [*:n], [n*] or CXSMILES labels ($_R1$, $X$); a row key R1 or X
names the position (a bare * is R). An R group is a SMILES fragment with its attachment point
([*], [*:1]), a fragment without one (attached at its first atom: "F", "OC", "c1ccccc1"), H, an
abbreviation or condensed formula from the fixed table below ("OMe", "iPr", "CH3", "N(CH3)2"),
or a substituted phenyl ("4-F-Ph", "3-OMe-phenyl"). A row whose fragment cannot be read gets no
structure (reported on stderr): its records still count in record_f1_by_label and are unlinked
in record_f1.

All YAML scalars are read as strings, so values keep their printed form ("12.30" stays "12.30").
The older key `compound` is accepted for `compound_id` (with a warning). A value that is a
legend key gets the legend text in `bin` (ignored by the scorer, kept for the deliverable).
A file that is not valid YAML is skipped with a warning (its documents score as missing); a
repeated records/rgroups/compounds block is merged and any other repeated key is reported. A
compound without a usable label is kept unlabeled (its structure still counts). Ambiguous labels
such as "NC" are refused rather than guessed; a leading dash of any width ("－OMe") is the bond.

Usage: python yaml_to_jsonl.py tables/*.yaml > pred.jsonl
"""

import json
import os
import re
import sys
import unicodedata

try:
    import yaml
except ImportError:
    sys.exit("yaml_to_jsonl.py needs PyYAML (pip install pyyaml)")
from rdkit import Chem, RDLogger

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score import _norm_label

RDLogger.DisableLog("rdApp.*")

_NULL = {"", "null", "~", "none", "None", "NULL", "Null"}
# Written out by hand rather than taken from rdAbbreviations.GetDefaultAbbreviations(): RDKit's
# list maps iBu and iPent to the sec-isomers, and a pinned table cannot change meaning between
# RDKit releases. '*' is the attachment point.
_ABBREVIATIONS = {
    # alkyl
    "Me": "*C", "Et": "*CC", "Pr": "*CCC", "nPr": "*CCC", "iPr": "*C(C)C", "Bu": "*CCCC", "nBu": "*CCCC",
    "iBu": "*CC(C)C", "sBu": "*C(C)CC", "tBu": "*C(C)(C)C", "nPent": "*CCCCC", "iPent": "*CCC(C)C",
    "nHex": "*CCCCCC", "cPr": "*C1CC1", "c-Pr": "*C1CC1", "cBu": "*C1CCC1", "cPent": "*C1CCCC1",
    "cHex": "*C1CCCCC1", "Cy": "*C1CCCCC1",
    # aryl, benzyl, acyl
    "Ph": "*c1ccccc1", "Bn": "*Cc1ccccc1", "Bz": "*C(=O)c1ccccc1", "Ac": "*C(C)=O", "CHO": "*C=O",
    # O / S / N substituents
    "OH": "*O", "OMe": "*OC", "OEt": "*OCC", "OiPr": "*OC(C)C", "OiBu": "*OCC(C)C", "OtBu": "*OC(C)(C)C",
    "OPh": "*Oc1ccccc1", "OBn": "*OCc1ccccc1", "OAc": "*OC(C)=O", "OCF3": "*OC(F)(F)F", "OCHF2": "*OC(F)F",
    "SH": "*S", "SMe": "*SC", "NH2": "*N", "NHMe": "*NC", "NMe": "*NC", "NHEt": "*NCC", "NMe2": "*N(C)C",
    "NEt2": "*N(CC)CC", "NHAc": "*NC(C)=O", "NO2": "*[N+](=O)[O-]", "NO": "*N=O", "CN": "*C#N",
    "NCF3": "*NC(F)(F)F", "N(OH)CH3": "*N(C)O",
    # halo-alkyl, carbonyl and sulfonyl groups
    "CF3": "*C(F)(F)F", "CHF2": "*C(F)F", "CH2F": "*CF", "CCl3": "*C(Cl)(Cl)Cl",
    "CO2H": "*C(=O)O", "COOH": "*C(=O)O", "CO2Me": "*C(=O)OC", "COOMe": "*C(=O)OC", "CO2Et": "*C(=O)OCC",
    "COOEt": "*C(=O)OCC", "CONH2": "*C(N)=O", "CONHMe": "*C(=O)NC", "SO2Me": "*S(C)(=O)=O", "Ms": "*S(C)(=O)=O",
    "SO2NH2": "*S(N)(=O)=O", "SO3H": "*S(=O)(=O)O", "Ts": "*S(=O)(=O)c1ccc(C)cc1",
    "Boc": "*C(=O)OC(C)(C)C", "Cbz": "*C(=O)OCc1ccccc1",
    # condensed formulas, common in CN/JP patent tables
    "CH3": "*C", "C2H5": "*CC", "CH2CH3": "*CC", "OCH3": "*OC", "OC2H5": "*OCC", "OCH2CH3": "*OCC",
    "SCH3": "*SC", "NHCH3": "*NC", "N(CH3)2": "*N(C)C", "CH2OH": "*CO", "CH2NH2": "*CN", "COCH3": "*C(C)=O",
    "CO2CH3": "*C(=O)OC", "COOCH3": "*C(=O)OC", "CO2C2H5": "*C(=O)OCC", "SO2CH3": "*S(C)(=O)=O",
    "NHCOCH3": "*NC(C)=O", "CH2Ph": "*Cc1ccccc1", "C6H5": "*c1ccccc1",
    # pseudohalides and acyl halides: valid SMILES as written, but with another meaning
    "SCN": "*SC#N", "NCS": "*N=C=S", "OCN": "*OC#N", "NCO": "*N=C=O", "COCl": "*C(=O)Cl",
}
# Labels with two common readings are refused rather than guessed ('NC': cyano drawn from the
# left, isocyano, or methylamino as SMILES).
_AMBIGUOUS = {"NC"}
_DASHES = "-‐‑‒–—−﹣－"
_PHENYL = re.compile(r"^([234])-([A-Za-z0-9()]+?)-?(?:Ph|phenyl|C6H4)$")
# Ring-closure label %99 is never used in the table above, so a substituent with its own ring
# ('4-cPr-Ph', '4-OPh-Ph') cannot close the phenyl ring early into a fused bicycle.
_RING_POSITION = {"2": "*c%99ccccc%99{}", "3": "*c%99cccc({})c%99", "4": "*c%99ccc({})cc%99"}
_ROW_META = {"compound_id", "compound", "label", "id", "smiles", "name"}
_LIST_KEYS = {"records", "rgroups", "compounds"}


class _Loader(yaml.BaseLoader):
    """BaseLoader (every scalar a string) that does not silently drop repeated keys: a repeated
    records/rgroups/compounds list is concatenated, any other repeated key is reported."""

    def __init__(self, stream):
        super().__init__(stream)
        self.notes = []

    def construct_mapping(self, node, deep=False):
        if not isinstance(node, yaml.MappingNode):
            return super().construct_mapping(node, deep=deep)
        mapping = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=True)
            value = self.construct_object(value_node, deep=True)
            if isinstance(key, (list, dict)):
                key = json.dumps(key)
            if key in mapping:
                line = key_node.start_mark.line + 1
                if key in _LIST_KEYS and isinstance(mapping[key], list) and isinstance(value, list):
                    mapping[key] = mapping[key] + value
                    self.notes.append(f"line {line}: repeated '{key}' block merged with the earlier one")
                    continue
                self.notes.append(f"line {line}: repeated key '{key}', the last value is kept")
            mapping[key] = value
        return mapping


def _null(x):
    return x is None or (isinstance(x, str) and x.strip() in _NULL)


def _canon(smiles):
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    return Chem.MolToSmiles(mol) if mol is not None else smiles


def _position_name(atom):
    """Row key that names this scaffold dummy: R<n> for [*:n], [n*], $_Rn$; the label for $X$; R for *."""
    n = atom.GetAtomMapNum() or atom.GetIsotope()
    if n:
        return f"R{n}"
    props = atom.GetPropsAsDict(includePrivate=True, includeComputed=False)
    label = str(props.get("atomLabel") or props.get("dummyLabel") or props.get("_MolFileRLabel") or "")
    label = label.lstrip("_")
    if label.isdigit():
        return f"R{label}"
    return "R" if label in ("", "*") else label


def _abbreviation(text):
    """Abbreviation, condensed formula or 'n-X-Ph' -> SMILES with '*', or None."""
    t = text
    if t in _ABBREVIATIONS:
        return _ABBREVIATIONS[t]
    m = _PHENYL.match(t)
    if m:
        sub = _ABBREVIATIONS.get(m.group(2))
        sub = sub[1:] if sub else (m.group(2) if m.group(2) in ("F", "Cl", "Br", "I") else None)
        if sub:
            return _RING_POSITION[m.group(1)].format(sub)
    return None


def _fragment(text, n):
    """R-group text -> mol with exactly one dummy atom, mapped to n; None if unreadable."""
    # '-F', '－OMe': the dash is the bond to the scaffold, not part of the group
    t = unicodedata.normalize("NFKC", str(text)).strip().lstrip(_DASHES).strip()
    if t in _AMBIGUOUS:
        return None
    if t in ("H", "[H]"):
        t = "[H][*]"
    t = _abbreviation(t) or t
    mol = Chem.MolFromSmiles(t)
    if mol is None:
        return None
    dummies = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 0]
    if len(dummies) > 1 or mol.GetNumAtoms() == len(dummies):
        return None
    rw = Chem.RWMol(mol)
    if not dummies:   # no attachment point written: attach at the first atom
        d = rw.AddAtom(Chem.Atom(0))
        rw.AddBond(0, d, Chem.BondType.SINGLE)
        dummies = [d]
    atom = rw.GetAtomWithIdx(dummies[0])
    atom.SetIsotope(0)
    atom.SetAtomMapNum(n)
    try:
        Chem.SanitizeMol(rw)
    except Exception:
        return None
    return rw.GetMol()


def enumerate_rows(scaffold_smiles, rows):
    """-> [(compound_id, smiles or None, problem or None)] for each R-group row."""
    scaffold = Chem.MolFromSmiles(scaffold_smiles) if not _null(scaffold_smiles) else None
    if scaffold is None:
        return [(_row_id(r), None, "scaffold missing or unreadable") for r in rows]
    positions = {}
    rw = Chem.RWMol(scaffold)
    for atom in rw.GetAtoms():
        if atom.GetAtomicNum() == 0:
            positions.setdefault(_position_name(atom), []).append(atom.GetIdx())
    # Every dummy gets its own map number; a label drawn twice (two R1) gets the same
    # substituent at both places.
    numbered, next_map = {}, 1
    for name, idxs in sorted(positions.items()):
        numbered[name] = []
        for idx in idxs:
            atom = rw.GetAtomWithIdx(idx)
            atom.SetIsotope(0)
            atom.SetAtomMapNum(next_map)
            numbered[name].append(next_map)
            next_map += 1
    base = rw.GetMol()
    out = []
    for row in rows:
        cid = _row_id(row)
        if not _null(row.get("smiles")):
            out.append((cid, str(row["smiles"]), None))
            continue
        given = {k: v for k, v in row.items() if k not in _ROW_META}
        missing = sorted(set(numbered) - set(given))
        unknown = sorted(set(given) - set(numbered))
        if missing or unknown:
            out.append((cid, None, f"R positions missing {missing} / not in scaffold {unknown}"))
            continue
        combo, bad = base, None
        for name, maps in numbered.items():
            for n in maps:
                frag = _fragment(given[name], n)
                if frag is None:
                    reason = "is ambiguous" if str(given[name]).strip() in _AMBIGUOUS else \
                        "is not a fragment or known abbreviation"
                    bad = f"{name} = {given[name]!r} {reason}"
                    break
                combo = Chem.CombineMols(combo, frag)
            if bad:
                break
        if bad:
            out.append((cid, None, bad))
            continue
        try:
            mol = Chem.RemoveHs(Chem.molzip(combo))
            out.append((cid, Chem.MolToSmiles(mol), None))
        except Exception as e:
            out.append((cid, None, f"molzip failed: {e}"))
    return out


def _row_id(row):
    for key in ("compound_id", "compound", "label", "id"):
        if not _null(row.get(key)):
            return str(row[key])
    return None


def _rows(table, key, where, warn):
    """table[key] as a list of mappings; anything else is reported, not silently dropped."""
    rows = table.get(key)
    if rows is None:
        return []
    if not isinstance(rows, list):
        warn(f"{where}: '{key}' is not a list, skipped")
        return []
    good = [r for r in rows if isinstance(r, dict)]
    if len(good) < len(rows):
        warn(f"{where}: {len(rows) - len(good)} entries of '{key}' are not mappings, skipped")
    return good


def convert(tables, warn=lambda msg: print("warning: " + msg, file=sys.stderr)):
    """Parsed records_yaml_v1 tables -> list of scorer documents (dicts), in first-seen doc order."""
    docs, renamed = {}, 0
    for t_index, table in enumerate(tables):
        if not isinstance(table, dict):
            warn(f"table {t_index}: not a mapping, skipped")
            continue
        head = table.get("table") if isinstance(table.get("table"), dict) else {}
        doc_id = head.get("doc_id") or table.get("doc_id")
        if _null(doc_id) or not isinstance(doc_id, str):
            warn(f"table {t_index}: no doc_id, or doc_id is not a string; skipped")
            continue
        where = f"{doc_id} {head.get('table_id') or ''}".strip()
        doc = docs.setdefault(doc_id, {"doc_id": doc_id, "compounds": [], "measurements": []})
        index = {_norm_label(c["id"]): c for c in doc["compounds"]}

        def add(cid, smiles):
            key = _norm_label(cid)
            if not key:   # kept: the scorer credits a structure without a label, it just links nothing
                warn(f"{where}: a compound without a usable compound_id ({cid!r}) is kept unlabeled")
                doc["compounds"].append({"id": cid, "smiles": smiles})
                return
            if key in index:
                old = index[key]
                if old["smiles"] is None and smiles:
                    old["smiles"] = smiles   # a later table could read what an earlier one could not
                elif smiles and old["smiles"] and _canon(smiles) != _canon(old["smiles"]):
                    warn(f"{where}: compound {cid} has two different structures; the first is kept")
                return
            index[key] = {"id": cid, "smiles": smiles}
            doc["compounds"].append(index[key])

        scaffold = table.get("scaffold") if isinstance(table.get("scaffold"), dict) else {}
        scaffold_smiles = next((scaffold[k] for k in ("cxsmiles", "smiles") if not _null(scaffold.get(k))), None)
        rows = _rows(table, "rgroups", where, warn)
        if rows:
            for cid, smiles, problem in enumerate_rows(scaffold_smiles, rows):
                if problem:
                    warn(f"{where}: compound {cid}: {problem}")
                add(cid, smiles)
        for c in _rows(table, "compounds", where, warn):
            add(_row_id(c), None if _null(c.get("smiles")) else str(c["smiles"]))
        legend = table.get("legend") if isinstance(table.get("legend"), dict) else {}
        for r in _rows(table, "records", where, warn):
            m = {k: v for k, v in r.items() if k != "footnote" and not _null(v)}
            if "compound_id" not in m and "compound" in m:
                m["compound_id"] = m.pop("compound")
                renamed += 1
            m.pop("compound", None)
            if "compound_id" not in m:
                m["compound_id"] = None   # no label printed: the record scores as unlinked
            value = str(m.get("value", "")).strip().rstrip("*")
            if value in legend:
                m["bin"] = legend[value]
            doc["measurements"].append(m)
    if renamed:
        warn(f"{renamed} records used the key 'compound'; read as 'compound_id'")
    return list(docs.values())


def load(paths, warn=lambda msg: print("warning: " + msg, file=sys.stderr)):
    """YAML files -> list of tables; a file that is not valid YAML is skipped with a warning."""
    tables = []
    for path in paths:
        try:
            with open(path, encoding="utf-8") as f:
                loader = _Loader(f.read())
            docs = []
            try:
                while loader.check_data():
                    docs.append(loader.get_data())
            finally:
                loader.dispose()
            tables.extend(docs)
            for note in loader.notes:
                warn(f"{path}: {note}")
        except (yaml.YAMLError, UnicodeDecodeError) as e:
            warn(f"{path}: not valid YAML, skipped ({getattr(e, 'problem', None) or e})")
    return tables


def main(paths):
    tables = load(paths)
    docs = convert(tables)
    n_c = sum(len(d["compounds"]) for d in docs)
    n_s = sum(1 for d in docs for c in d["compounds"] if c["smiles"])
    n_m = sum(len(d["measurements"]) for d in docs)
    for d in docs:
        print(json.dumps(d, ensure_ascii=False))
    print(f"{len(tables)} tables -> {len(docs)} documents, {n_c} compounds ({n_s} with a structure), "
          f"{n_m} records", file=sys.stderr)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1:])
