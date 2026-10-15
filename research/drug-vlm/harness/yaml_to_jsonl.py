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
([*], [*:1]), a fragment without one (attached at its first atom: "F", "OC", "c1ccccc1"), H, or
an abbreviation ("OMe", "CF3", "iPr", "CO2Et", RDKit's default list plus a few more). A row
whose fragment cannot be read gets no structure (reported on stderr): its records still count
in record_f1_by_label and are unlinked in record_f1.

All YAML scalars are read as strings, so values keep their printed form ("12.30" stays "12.30").
The older key `compound` is accepted for `compound_id` (with a warning). A value that is a
legend key gets the legend text in `bin` (ignored by the scorer, kept for the deliverable).

Usage: python yaml_to_jsonl.py tables/*.yaml > pred.jsonl
"""

import json
import os
import sys

try:
    import yaml
except ImportError:
    sys.exit("yaml_to_jsonl.py needs PyYAML (pip install pyyaml)")
from rdkit import Chem, RDLogger
from rdkit.Chem import rdAbbreviations

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score import _norm_label

RDLogger.DisableLog("rdApp.*")

_NULL = {"", "null", "~", "none", "None", "NULL", "Null"}
_ABBREVIATIONS = {a.label: Chem.MolToSmiles(a.mol) for a in rdAbbreviations.GetDefaultAbbreviations()}
_ABBREVIATIONS.update({
    "Me": "*C", "Pr": "*CCC", "Bu": "*CCCC", "cPr": "*C1CC1", "cBu": "*C1CCC1", "cPent": "*C1CCCC1",
    "cHex": "*C1CCCCC1", "Ph": "*c1ccccc1", "Bn": "*Cc1ccccc1", "OPh": "*Oc1ccccc1", "OBn": "*OCc1ccccc1",
    "OH": "*O", "SH": "*S", "NH2": "*N", "NHMe": "*NC", "NMe2": "*N(C)C", "OCF3": "*OC(F)(F)F",
    "OCHF2": "*OC(F)F", "CHF2": "*C(F)F", "CH2F": "*CF", "CONH2": "*C(N)=O", "CONHMe": "*C(=O)NC",
    "SO2Me": "*S(C)(=O)=O", "SO2NH2": "*S(N)(=O)=O", "CO2Me": "*C(=O)OC", "COOMe": "*C(=O)OC",
    "Boc": "*C(=O)OC(C)(C)C", "Cbz": "*C(=O)OCc1ccccc1", "Ms": "*S(C)(=O)=O", "Ts": "*S(=O)(=O)c1ccc(C)cc1",
})
_ROW_META = {"compound_id", "compound", "label", "id", "smiles", "name"}


def _null(x):
    return x is None or (isinstance(x, str) and x.strip() in _NULL)


def _position_name(atom):
    """Row key that names this scaffold dummy: R<n> for [*:n], [n*], $_Rn$; the label for $X$."""
    n = atom.GetAtomMapNum() or atom.GetIsotope()
    if n:
        return f"R{n}"
    props = atom.GetPropsAsDict(includePrivate=True, includeComputed=False)
    label = str(props.get("atomLabel") or props.get("dummyLabel") or props.get("_MolFileRLabel") or "")
    label = label.lstrip("_")
    if label.isdigit():
        return f"R{label}"
    return label or "R"


def _fragment(text, n):
    """R-group text -> mol with exactly one dummy atom, mapped to n; None if unreadable."""
    t = str(text).strip()
    if t in ("H", "[H]", "-H"):
        t = "[H][*]"
    t = _ABBREVIATIONS.get(t.lstrip("-"), t)
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
    out = []
    if scaffold is None:
        return [(_row_id(r), None, "scaffold missing or unreadable") for r in rows]
    positions = {}
    rw = Chem.RWMol(scaffold)
    for atom in rw.GetAtoms():
        if atom.GetAtomicNum() == 0:
            positions.setdefault(_position_name(atom), []).append(atom.GetIdx())
    # Every dummy gets its own map number; a label drawn twice (two R1) gets the same
    # substituent at both places.
    numbered = {}
    next_map = 1
    for name, idxs in sorted(positions.items()):
        numbered[name] = []
        for idx in idxs:
            atom = rw.GetAtomWithIdx(idx)
            atom.SetIsotope(0)
            atom.SetAtomMapNum(next_map)
            numbered[name].append(next_map)
            next_map += 1
    base = rw.GetMol()
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
                    bad = f"{name} = {given[name]!r} is not a fragment or known abbreviation"
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


def convert(tables, warn=lambda msg: print("warning: " + msg, file=sys.stderr)):
    """Parsed records_yaml_v1 tables -> list of scorer documents (dicts), in first-seen doc order."""
    docs, renamed = {}, 0
    for t_index, table in enumerate(tables):
        if not isinstance(table, dict):
            warn(f"table {t_index}: not a mapping, skipped")
            continue
        head = table.get("table") if isinstance(table.get("table"), dict) else {}
        doc_id = head.get("doc_id") or table.get("doc_id")
        if _null(doc_id):
            warn(f"table {t_index}: no doc_id, skipped")
            continue
        where = f"{doc_id} {head.get('table_id') or ''}".strip()
        doc = docs.setdefault(doc_id, {"doc_id": doc_id, "compounds": [], "measurements": []})
        seen = {_norm_label(c["id"]): c["smiles"] for c in doc["compounds"]}

        def add(cid, smiles):
            key = _norm_label(cid)
            if not key:
                return
            if key in seen:
                if smiles and seen[key] and Chem.CanonSmiles(smiles) != Chem.CanonSmiles(seen[key]):
                    warn(f"{where}: compound {cid} has two different structures; the first is kept")
                return
            seen[key] = smiles
            doc["compounds"].append({"id": cid, "smiles": smiles})

        scaffold = table.get("scaffold") if isinstance(table.get("scaffold"), dict) else {}
        rows = [r for r in (table.get("rgroups") or []) if isinstance(r, dict)]
        if rows:
            for cid, smiles, problem in enumerate_rows(scaffold.get("cxsmiles") or scaffold.get("smiles"), rows):
                if problem:
                    warn(f"{where}: compound {cid}: {problem}")
                add(cid, smiles)
        for c in table.get("compounds") or []:
            if isinstance(c, dict):
                add(_row_id(c), None if _null(c.get("smiles")) else str(c["smiles"]))
        legend = table.get("legend") if isinstance(table.get("legend"), dict) else {}
        for r in table.get("records") or []:
            if not isinstance(r, dict):
                continue
            m = {k: v for k, v in r.items() if k != "footnote" and not _null(v)}
            if "compound_id" not in m and "compound" in m:
                m["compound_id"] = m.pop("compound")
                renamed += 1
            value = str(m.get("value", "")).strip().rstrip("*")
            if value in legend:
                m["bin"] = legend[value]
            doc["measurements"].append(m)
    if renamed:
        warn(f"{renamed} records used the key 'compound'; read as 'compound_id'")
    return list(docs.values())


def main(paths):
    tables = []
    for path in paths:
        with open(path, encoding="utf-8") as f:
            tables.extend(yaml.load_all(f, Loader=yaml.BaseLoader))
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
