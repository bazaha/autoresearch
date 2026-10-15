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
        (k1, s1), (k2, s2) = score.structure_keys(L_ALA), score.structure_keys(D_ALA)
        self.assertNotEqual(k1, k2)
        self.assertEqual(s1, s2)

    def test_invalid_smiles(self):
        self.assertIsNone(score.structure_key("C1CC"))
        self.assertIsNone(score.structure_key(""))
        self.assertIsNone(score.structure_key(None))

    def test_markush_dummy_atoms_use_cxsmiles(self):
        k1, k2 = score.structure_key("*c1ccccc1"), score.structure_key("c1ccc(*)cc1")
        self.assertTrue(k1.startswith("MARKUSH:"))
        self.assertEqual(k1, k2)

    def test_r_label_conventions_are_equivalent(self):
        spellings = ["[1*]c1ccccc1", "[*:1]c1ccccc1", "*c1ccccc1 |$R1;;;;;;$|", "*c1ccccc1 |$_R1;;;;;;$|"]
        keys = {score.structure_key(s) for s in spellings}
        self.assertEqual(len(keys), 1)
        self.assertNotEqual(keys.pop(), score.structure_key("[2*]c1ccccc1"))

    def test_markush_position_variation(self):
        cl_any = "*Cl.c1ccccc1 |m:0:2.3.4.5.6.7|"
        self.assertEqual(score.structure_key(cl_any), score.structure_key("c1ccccc1.Cl* |m:7:0.1.2.3.4.5|"))
        others = ["*Br.c1ccccc1 |m:0:2.3.4.5.6.7|", "c1ccccc1", "Clc1ccccc1"]
        self.assertEqual(len({score.structure_key(s) for s in [cl_any] + others}), 4)
        # pyridine: Cl at para/meta (either side, symmetric) vs meta/ortho
        para_meta = score.structure_key("*Cl.c1ccncc1 |m:0:2.3|")
        self.assertEqual(para_meta, score.structure_key("*Cl.c1ccncc1 |m:0:2.7|"))
        self.assertNotEqual(para_meta, score.structure_key("*Cl.c1ccncc1 |m:0:3.4|"))
        self.assertFalse(score.is_multifragment(cl_any))

    def test_atom_maps_and_labels_on_real_atoms_are_ignored(self):
        self.assertEqual(score.structure_key("[CH3:1][C:2](=O)O"), score.structure_key("CC(=O)O"))
        self.assertEqual(score.structure_key("[1*]c1ccc(C)cc1 |$;;;;Me;;;$|"), score.structure_key("[1*]c1ccc(C)cc1"))

    def test_enhanced_stereo(self):
        rac_a, rac_b = "C[C@H](N)C(=O)O |&1:1|", "C[C@@H](N)C(=O)O |&1:1|"
        self.assertEqual(score.structure_key(rac_a), score.structure_key(rac_b))  # same racemate
        self.assertNotEqual(score.structure_key(rac_a), score.structure_key(L_ALA))  # racemate != enantiomer
        self.assertEqual(score.structure_keys(rac_a)[1], score.structure_keys(L_ALA)[1])

    def test_multifragment_flag_ignores_salts_and_solvents(self):
        self.assertFalse(score.is_multifragment(ASPIRIN_NA_SALT))
        self.assertFalse(score.is_multifragment(ASPIRIN + ".CCO"))
        self.assertTrue(score.is_multifragment(ASPIRIN + ".CCCCCCCC"))

    def test_oversized_structure_is_invalid(self):
        self.assertIsNone(score.structure_key("C" * (score.MAX_HEAVY_ATOMS + 1)))


class MeasurementTests(unittest.TestCase):
    def test_unit_conversion(self):
        a = score.normalize_measurement(meas("1", 0.0123, unit="µM"))
        b = score.normalize_measurement(meas("1", 12.3, unit="nM"))
        self.assertEqual(a[:5], b[:5])
        self.assertAlmostEqual(a[5][0], b[5][0])

    def test_p_value_conversion_flips_relation(self):
        a = score.normalize_measurement(meas("1", 7.0, unit="", endpoint="pIC50", relation=">"))
        self.assertEqual(a[0], "ic50")
        self.assertEqual(a[2], "<")
        self.assertAlmostEqual(a[5][0], 100.0)

    def test_relation_and_endpoint_normalization(self):
        a = score.normalize_measurement(meas("1", 10, unit="uM", endpoint="IC₅₀", relation="≥"))
        b = score.normalize_measurement(meas("1", 10, unit="μM", endpoint="ic50", relation=">"))
        self.assertEqual(a, b)

    def test_non_numeric_value(self):
        self.assertIsNone(score.normalize_measurement(meas("1", "n.d.")))

    def test_unit_inside_endpoint_name_is_ignored(self):
        a = score.normalize_measurement(meas("1", 1.2, unit="uM", endpoint="IC50 (µM)"))
        b = score.normalize_measurement(meas("1", 1200, unit="nM", endpoint="IC50"))
        self.assertEqual(a[:5], b[:5])

    def test_compound_units_canonicalize(self):
        self.assertEqual(score._norm_unit("mL·min⁻¹·kg⁻¹"), score._norm_unit("mL/min/kg"))
        self.assertEqual(score._norm_unit("mL min-1 kg-1"), "ml/min/kg")
        self.assertEqual(score._norm_unit("µM"), "um")

    def test_single_dose_condition_must_match(self):
        at_1um = dict(meas("1", 87, unit="%", endpoint="% inhibition"), conc=1, conc_unit="uM")
        at_1000nm = dict(at_1um, conc=1000, conc_unit="nM")
        at_10um = dict(at_1um, conc=10)
        self.assertEqual(score.normalize_measurement(at_1um), score.normalize_measurement(at_1000nm))
        self.assertNotEqual(score.normalize_measurement(at_1um), score.normalize_measurement(at_10um))

    def test_px_vs_converted_molar_allows_rounding_only(self):
        gold = score.normalize_measurement(meas("1", 7.2, unit="", endpoint="pIC50"))
        rounded = score.normalize_measurement(meas("1", 63, unit="nM"))
        self.assertEqual(gold[:5], rounded[:5])
        self.assertTrue(score._close(gold[5:], rounded[5:]))
        same_scale_typo = score.normalize_measurement(meas("1", 12.5, unit="nM"))
        same_scale_gold = score.normalize_measurement(meas("1", 12.3, unit="nM"))
        self.assertFalse(score._close(same_scale_gold[5:], same_scale_typo[5:]))

    def test_ranges(self):
        a = score.normalize_measurement(meas("1", [10, 30], unit="uM"))
        b = score.normalize_measurement(meas("1", [30000, 10000], unit="nM"))
        c = score.normalize_measurement(meas("1", 10, unit="uM"))
        self.assertEqual(a[2], "range")
        self.assertEqual(a[:5], b[:5])
        self.assertTrue(score._close(a[5:], b[5:]))
        self.assertNotEqual(a[:5], c[:5])


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

    def test_duplicate_records_are_false_positives(self):
        pred = doc(
            [{"id": "12a", "smiles": ASPIRIN}, {"id": "12b", "smiles": PARACETAMOL}],
            [meas("12a", 12.3), meas("12a", 12.3), meas("12b", 450.0)],
        )
        s = score.aggregate([score.score_document(self.gold, pred)])
        self.assertAlmostEqual(s["record_precision"], 2 / 3)
        self.assertEqual(s["record_recall"], 1.0)

    def test_joined_compound_ids_get_no_credit(self):
        pred = doc(
            [{"id": "12a", "smiles": ASPIRIN}, {"id": "12b", "smiles": PARACETAMOL}],
            [meas("12a;12b", 12.3)],
        )
        s = score.aggregate([score.score_document(self.gold, pred)])
        self.assertEqual(s["record_f1"], 0.0)
        self.assertEqual(s["orphan_record_rate"], 1.0)

    def test_unit_mismatch_and_flipped_qualifier_are_wrong(self):
        pred = doc(
            [{"id": "12a", "smiles": ASPIRIN}, {"id": "12b", "smiles": PARACETAMOL}],
            [meas("12a", 12.3, unit="uM"), meas("12b", 450.0, relation="<")],
        )
        s = score.aggregate([score.score_document(self.gold, pred)])
        self.assertEqual(s["record_f1"], 0.0)

    def test_id_link_detects_swapped_labels(self):
        pred = doc(
            [{"id": "12b", "smiles": ASPIRIN}, {"id": "12a", "smiles": PARACETAMOL}],
            [],
        )
        s = score.aggregate([score.score_document(self.gold, pred)], scalar="struct_f1")
        self.assertEqual(s["struct_f1"], 1.0)
        self.assertEqual(s["id_link_f1"], 0.0)

    def test_prediction_caps(self):
        old = score.MAX_PRED_COMPOUNDS
        try:
            score.MAX_PRED_COMPOUNDS = 2
            pred = doc([{"id": "12a", "smiles": ASPIRIN}, {"id": "12b", "smiles": PARACETAMOL},
                        {"id": "x", "smiles": ASPIRIN}])
            row = score.score_document(self.gold, pred)
        finally:
            score.MAX_PRED_COMPOUNDS = old
        self.assertEqual(row["n_pred_struct"], 3)
        self.assertEqual(row["n_invalid_struct"], 1)
        self.assertEqual(row["struct_tp"], 2)

    def test_explicit_scalar(self):
        pred = doc([{"id": "12a", "smiles": ASPIRIN}, {"id": "12b", "smiles": PARACETAMOL}], [])
        rows = [score.score_document(self.gold, pred)]
        self.assertEqual(score.aggregate(rows)["scalar"], "composite")
        self.assertEqual(score.aggregate(rows)["score"], 0.5)
        self.assertEqual(score.aggregate(rows, scalar="struct_f1")["score"], 1.0)

    def test_unscorable_gold_does_not_penalize_precision(self):
        gold = doc([{"id": "1", "smiles": ASPIRIN}, {"id": "2", "smiles": "not-a-smiles"}])
        pred = doc([{"id": "1", "smiles": ASPIRIN}, {"id": "2", "smiles": PARACETAMOL}])
        s = score.aggregate([score.score_document(gold, pred)])
        self.assertEqual(s["struct_precision"], 1.0)
        self.assertEqual(s["n_gold_unscorable"], 1)

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


class FileScoringTests(unittest.TestCase):
    def test_bad_prediction_lines_and_missing_docs_score_as_empty(self):
        import json, os, tempfile
        with tempfile.TemporaryDirectory() as tmp:
            gold_path, pred_path = os.path.join(tmp, "gold.jsonl"), os.path.join(tmp, "pred.jsonl")
            with open(gold_path, "w") as f:
                for i in range(3):
                    f.write(json.dumps(doc([{"smiles": ASPIRIN}], doc_id=f"d{i}")) + "\n")
            with open(pred_path, "w") as f:
                f.write(json.dumps(doc([{"smiles": ASPIRIN}], doc_id="d0")) + "\n")
                f.write(json.dumps(doc([{"smiles": PARACETAMOL}], doc_id="d0")) + "\n")  # later duplicate ignored
                f.write("{not json\n")
            per_doc, missing = score.score_files(gold_path, pred_path)
        self.assertEqual(missing, 2)
        self.assertAlmostEqual(score.aggregate(per_doc)["struct_recall"], 1 / 3)
        self.assertEqual(score.aggregate(per_doc)["struct_precision"], 1.0)


class StatisticsTests(unittest.TestCase):
    def _per_doc(self, n_correct, n_total, family_size=1):
        rows = []
        for i in range(n_total):
            gold = dict(doc([{"smiles": ASPIRIN}], doc_id=f"d{i}"), family_id=f"f{i // family_size}")
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

    def test_clustered_bootstrap_is_wider(self):
        lo1, hi1 = score.bootstrap_ci(self._per_doc(70, 100))
        # same accuracy, but correct/incorrect docs come in families of 10
        clustered = self._per_doc(70, 100, family_size=10)
        lo2, hi2 = score.bootstrap_ci(clustered)
        self.assertGreater(hi2 - lo2, hi1 - lo1)
        self.assertEqual(score.paired_compare(clustered, clustered)["n_clusters"], 10)

    def test_rdkit_is_pinned_version(self):
        self.assertEqual(score.RDKIT_VERSION, score.RDKIT_PIN)

    def test_seed_pooling(self):
        a, b = self._per_doc(60, 100), self._per_doc(80, 100)
        pooled = score.pool_seeds([a, b])
        self.assertEqual(len(pooled), 100)
        self.assertAlmostEqual(score.aggregate(pooled)["score"], 0.70)

    def test_paired_compare_identical_runs(self):
        rows = self._per_doc(50, 100)
        r = score.paired_compare(rows, rows)
        self.assertEqual(r["delta"], 0.0)
        self.assertEqual(r["p_not_better"], 1.0)


if __name__ == "__main__":
    unittest.main()
