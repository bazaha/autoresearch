"""Unit tests for score.py. Run: python -m unittest test_score.py"""

import unittest

import score

ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
ASPIRIN_NA_SALT = "CC(=O)Oc1ccccc1C(=O)[O-].[Na+]"
L_ALA = "C[C@H](N)C(=O)O"
D_ALA = "C[C@@H](N)C(=O)O"
PARACETAMOL = "CC(=O)Nc1ccc(O)cc1"


def doc(compounds=(), measurements=(), doc_id="d1"):
    return {"doc_id": doc_id, "compounds": list(compounds), "measurements": list(measurements)}


def meas(cid, value, unit="nM", endpoint="IC50", target="EGFR", relation="="):
    return {"compound_id": cid, "value": value, "unit": unit, "endpoint": endpoint,
            "target": target, "relation": relation}


class StructureKeyTests(unittest.TestCase):
    def test_equivalent_smiles_match(self):
        self.assertEqual(score.structure_key("OCC"), score.structure_key("CCO"))

    def test_salt_stripped_to_parent(self):
        self.assertEqual(score.structure_key(ASPIRIN_NA_SALT), score.structure_key(ASPIRIN))

    def test_stereo_distinguished_but_same_skeleton(self):
        k1, k2 = score.structure_key(L_ALA), score.structure_key(D_ALA)
        self.assertNotEqual(k1, k2)
        self.assertEqual(score.skeleton_key(k1), score.skeleton_key(k2))

    def test_invalid_smiles(self):
        self.assertIsNone(score.structure_key("C1CC"))
        self.assertIsNone(score.structure_key(""))
        self.assertIsNone(score.structure_key(None))

    def test_markush_dummy_atoms_use_cxsmiles(self):
        k1, k2 = score.structure_key("*c1ccccc1"), score.structure_key("c1ccc(*)cc1")
        self.assertTrue(k1.startswith("CXSMILES:"))
        self.assertEqual(k1, k2)


class MeasurementTests(unittest.TestCase):
    def test_unit_conversion(self):
        a = score.normalize_measurement(meas("1", 0.0123, unit="µM"))
        b = score.normalize_measurement(meas("1", 12.3, unit="nM"))
        self.assertEqual(a[:4], b[:4])
        self.assertAlmostEqual(a[4], b[4])

    def test_p_value_conversion_flips_relation(self):
        a = score.normalize_measurement(meas("1", 7.0, unit="", endpoint="pIC50", relation=">"))
        self.assertEqual(a[0], "ic50")
        self.assertEqual(a[2], "<")
        self.assertAlmostEqual(a[4], 100.0)

    def test_relation_and_endpoint_normalization(self):
        a = score.normalize_measurement(meas("1", 10, unit="uM", endpoint="IC₅₀", relation="≥"))
        b = score.normalize_measurement(meas("1", 10, unit="μM", endpoint="ic50", relation=">"))
        self.assertEqual(a, b)

    def test_non_numeric_value(self):
        self.assertIsNone(score.normalize_measurement(meas("1", "n.d.")))


class DocumentScoringTests(unittest.TestCase):
    def setUp(self):
        self.gold = doc(
            [{"id": "12a", "smiles": ASPIRIN}, {"id": "12b", "smiles": PARACETAMOL}],
            [meas("12a", 12.3), meas("12b", 450.0)],
        )

    def test_perfect_prediction(self):
        pred = doc(
            [{"id": "Compound 12a", "smiles": ASPIRIN_NA_SALT}, {"id": "12b", "smiles": PARACETAMOL}],
            [meas("12a", 0.0123, unit="uM"), meas("12b", 0.45, unit="µM")],
        )
        s = score.aggregate([score.score_document(self.gold, pred)])
        self.assertEqual(s["struct_f1"], 1.0)
        self.assertEqual(s["record_f1"], 1.0)
        self.assertEqual(s["score"], 1.0)

    def test_swapped_values_break_records_not_structures(self):
        pred = doc(
            [{"id": "12a", "smiles": ASPIRIN}, {"id": "12b", "smiles": PARACETAMOL}],
            [meas("12a", 450.0), meas("12b", 12.3)],
        )
        s = score.aggregate([score.score_document(self.gold, pred)])
        self.assertEqual(s["struct_f1"], 1.0)
        self.assertEqual(s["record_f1"], 0.0)
        self.assertEqual(s["score"], 0.5)

    def test_wrong_structure_breaks_record_but_not_label_diagnostic(self):
        pred = doc(
            [{"id": "12a", "smiles": "CCO"}, {"id": "12b", "smiles": PARACETAMOL}],
            [meas("12a", 12.3), meas("12b", 450.0)],
        )
        s = score.aggregate([score.score_document(self.gold, pred)])
        self.assertAlmostEqual(s["record_f1"], 0.5)
        self.assertEqual(s["record_f1_by_label"], 1.0)

    def test_invalid_and_orphans_counted(self):
        pred = doc([{"id": "12a", "smiles": "C1CC"}], [meas("99", 1.0)])
        s = score.aggregate([score.score_document(self.gold, pred)])
        self.assertEqual(s["invalid_smiles_rate"], 1.0)
        self.assertEqual(s["orphan_record_rate"], 1.0)
        self.assertEqual(s["score"], 0.0)

    def test_empty_prediction(self):
        s = score.aggregate([score.score_document(self.gold, {})])
        self.assertEqual(s["score"], 0.0)

    def test_ocsr_crop_only_uses_structure_f1(self):
        gold = doc([{"id": None, "smiles": L_ALA}])
        good = score.aggregate([score.score_document(gold, doc([{"smiles": L_ALA}]))])
        stereo_wrong = score.aggregate([score.score_document(gold, doc([{"smiles": D_ALA}]))])
        self.assertEqual(good["score"], 1.0)
        self.assertEqual(stereo_wrong["score"], 0.0)
        self.assertEqual(stereo_wrong["struct_f1_nostereo"], 1.0)
        self.assertEqual(stereo_wrong["struct_tanimoto"], 1.0)  # Morgan FP ignores chirality


class StatisticsTests(unittest.TestCase):
    def _per_doc(self, n_correct, n_total):
        rows = []
        for i in range(n_total):
            gold = doc([{"smiles": ASPIRIN}], doc_id=f"d{i}")
            pred = doc([{"smiles": ASPIRIN if i < n_correct else PARACETAMOL}], doc_id=f"d{i}")
            rows.append(score.score_document(gold, pred))
        return rows

    def test_bootstrap_is_deterministic_and_brackets_estimate(self):
        rows = self._per_doc(70, 100)
        self.assertEqual(score.bootstrap_ci(rows), score.bootstrap_ci(rows))
        lo, hi = score.bootstrap_ci(rows)
        self.assertLess(lo, 0.70)
        self.assertGreater(hi, 0.70)

    def test_paired_compare_detects_clear_improvement(self):
        base, new = self._per_doc(60, 200), self._per_doc(80, 200)
        r = score.paired_compare(base, new)
        self.assertAlmostEqual(r["delta"], 0.10)
        self.assertGreater(r["ci_lo"], 0.0)
        self.assertLess(r["p_not_better"], 0.01)

    def test_paired_compare_identical_runs(self):
        rows = self._per_doc(50, 100)
        r = score.paired_compare(rows, rows)
        self.assertEqual(r["delta"], 0.0)
        self.assertEqual(r["p_not_better"], 1.0)


if __name__ == "__main__":
    unittest.main()
