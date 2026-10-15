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

    def test_metal_complexes_are_keyed_as_drawn(self):
        cisplatin = score.structure_key("N.N.Cl[Pt]Cl")
        self.assertEqual(cisplatin, score.structure_key("[NH3+][Pt-2]([NH3+])(Cl)Cl"))
        self.assertNotEqual(cisplatin, score.structure_key("N"))

    def test_free_metal_counterions_are_salts(self):
        zn_salt = "CC(=O)Oc1ccccc1C(=O)[O-].CC(=O)Oc1ccccc1C(=O)[O-].[Zn+2]"
        self.assertEqual(score.structure_key(zn_salt), score.structure_key(ASPIRIN))

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

    def test_missing_value(self):
        self.assertIsNone(score.normalize_measurement(meas("1", None)))
        self.assertIsNone(score.normalize_measurement(meas("1", "")))

    def test_categorical_values(self):
        nd = score.normalize_measurement(meas("1", "n.d."))
        self.assertEqual(nd[2], "cat:nd")
        self.assertEqual(nd, score.normalize_measurement(meas("1", "N.D.", unit="")))
        self.assertNotEqual(score.normalize_measurement(meas("1", "+++"))[2],
                            score.normalize_measurement(meas("1", "++"))[2])

    def test_value_strings(self):
        gt = score.normalize_measurement(meas("1", ">10", unit="uM"))
        self.assertEqual((gt[2], gt[5]), (">", (10000.0,)))
        rng = score.normalize_measurement(meas("1", "10–30", unit="nM"))
        self.assertEqual((rng[2], rng[5]), ("range", (10.0, 30.0)))
        self.assertEqual(score.normalize_measurement(meas("1", "1,200"))[5], (1200.0,))
        self.assertEqual(score.normalize_measurement(meas("1", "−0.5", unit="", endpoint="logD"))[5], (-0.5,))
        self.assertEqual(score.normalize_measurement(meas("1", "1e-5", unit="M"))[5], (1e-5 * 1e9,))

    def test_value_strings_with_error_exponent_and_unit(self):
        ref = score.normalize_measurement(meas("1", 12.0))
        self.assertEqual(score.normalize_measurement(meas("1", "12 ± 3"))[:6], ref[:6])
        self.assertEqual(score.normalize_measurement(meas("1", "12 (n=3)"))[:6], ref[:6])
        sci = score.normalize_measurement(meas("1", "1.2 × 10⁻⁵", unit="M"))
        self.assertAlmostEqual(sci[5][0], 12000.0)
        ten_um = score.normalize_measurement(meas("1", 10, unit="uM"))
        self.assertEqual(score.normalize_measurement(meas("1", "10 µM", unit=""))[:6], ten_um[:6])
        # unit only in the column header
        self.assertEqual(score.normalize_measurement(meas("1", 10, unit="", endpoint="IC50 (µM)"))[:6], ten_um[:6])

    def test_more_unit_and_endpoint_spellings(self):
        a = score.normalize_measurement(meas("1", 5, unit="µmol/L"))
        b = score.normalize_measurement(meas("1", 5, unit="uM"))
        self.assertEqual(a, b)
        self.assertEqual(score._norm_endpoint("Kᵢ"), score._norm_endpoint("Ki"))
        self.assertEqual(score._norm_endpoint("T½"), "t1/2")
        self.assertEqual(score._norm_endpoint("half-life"), "t1/2")
        px = score.normalize_measurement(meas("1", 7, unit="M", endpoint="pIC50"))
        self.assertEqual((px[0], px[3]), ("ic50", "molar"))

    def test_concentration_written_in_endpoint(self):
        a = score.normalize_measurement(meas("1", 87, unit="%", endpoint="% inh @ 1 µM"))
        b = dict(meas("1", 87, unit="%", endpoint="% inhibition"), conc=1, conc_unit="uM")
        self.assertEqual(a, score.normalize_measurement(b))

    def test_mass_concentration_uses_structure_mw(self):
        mw = score.mol_weight(ASPIRIN)
        gold = score.normalize_measurement(meas("1", 1.8, unit="ug/mL", endpoint="MIC"), mw)
        pred = score.normalize_measurement(meas("1", 10, unit="uM", endpoint="MIC"), mw)
        self.assertEqual(gold[:5], pred[:5])
        self.assertTrue(score._close(gold[5:], pred[5:]))
        no_mw = score.normalize_measurement(meas("1", 1.8, unit="ug/mL", endpoint="MIC"))
        self.assertEqual(no_mw[3], "dim:mass+1.vol-1")   # without a structure: compared as mass/volume
        self.assertEqual(no_mw[:6], score.normalize_measurement(meas("1", 1.8, unit="mg/L", endpoint="MIC"))[:6])

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

    def test_converted_value_must_round_to_gold_as_printed(self):
        gold = score.normalize_measurement(meas("1", "50", unit="nM"))
        misread = score.normalize_measurement(meas("1", 7.26, unit="", endpoint="pIC50"))  # = 55 nM
        faithful = score.normalize_measurement(meas("1", 7.30, unit="", endpoint="pIC50"))  # = 50.1 nM
        self.assertFalse(score._close(gold[5:], misread[5:]))
        self.assertTrue(score._close(gold[5:], faithful[5:]))
        # a coarsely printed prediction does not widen the tolerance
        coarse = score.normalize_measurement(meas("1", 7, unit="", endpoint="pIC50"))
        self.assertFalse(score._close(gold[5:], coarse[5:]))
        # trailing zeros printed in the gold are kept when the value is a string
        self.assertEqual(score._half_unit("1.80"), 0.005)

    def test_ranges(self):
        a = score.normalize_measurement(meas("1", [10, 30], unit="uM"))
        b = score.normalize_measurement(meas("1", [30000, 10000], unit="nM"))
        c = score.normalize_measurement(meas("1", 10, unit="uM"))
        self.assertEqual(a[2], "range")
        self.assertEqual(a[:5], b[:5])
        self.assertTrue(score._close(a[5:], b[5:]))
        self.assertNotEqual(a[:5], c[:5])


def record_tp(gold_m, pred_m):
    """record_tp for one aspirin record on each side; fields are passed straight through."""
    gold = doc([{"id": "1", "smiles": ASPIRIN}], [dict(compound_id="1", **gold_m)])
    pred = doc([{"id": "1", "smiles": ASPIRIN}], [dict(compound_id="1", **pred_m)])
    return score.score_document(gold, pred)["record_tp"]


class RecordKeyV2Tests(unittest.TestCase):
    def test_chinese_and_greek_labels_are_kept(self):
        self.assertEqual(record_tp(dict(endpoint="抑制率", value=50, unit="%", target="EGFR"),
                                   dict(endpoint="清除率", value=50, unit="%", target="EGFR")), 0)
        self.assertEqual(record_tp(dict(endpoint="IC50", value=5, unit="nM", target="PI3Kα"),
                                   dict(endpoint="IC50", value=5, unit="nM", target="PI3Kβ")), 0)
        self.assertEqual(record_tp(dict(endpoint="半衰期", value=2, unit="h", target=""),
                                   dict(endpoint="t1/2", value=120, unit="min", target="")), 1)
        self.assertEqual(record_tp(dict(endpoint="抑制率", value=50, unit="%", target="EGFR"),
                                   dict(endpoint="% inhibition", value=50, unit="%", target="EGFR")), 1)

    def test_label_that_normalizes_to_nothing_fails_closed(self):
        self.assertIsNone(score.normalize_measurement(meas("1", 5, endpoint="★★")))
        self.assertIsNone(score.normalize_measurement(meas("1", 5, target="—")))

    def test_gold_defined_context_fields(self):
        base = dict(endpoint="CLint", value=12, unit="uL/min/mg", target="")
        human, rat = dict(base, species="human", matrix="liver microsomes"), dict(base, species="rat", matrix="liver microsomes")
        self.assertEqual(record_tp(human, rat), 0)
        self.assertEqual(record_tp(human, dict(base, matrix="HLM")), 1)            # HLM implies human
        self.assertEqual(record_tp(human, dict(base, endpoint="CLint (HLM)")), 1)  # context in the header
        self.assertEqual(record_tp(human, base), 0)                                # gold has species, pred omits it
        self.assertEqual(record_tp(base, human), 1)                                # pred adds a field gold lacks
        ic50 = dict(endpoint="IC50", value=5, unit="nM", target="EGFR")
        self.assertEqual(record_tp(dict(ic50, assay_type="biochemical"), dict(ic50, assay_type="cell")), 0)
        self.assertEqual(record_tp(dict(ic50, assay_type="enzymatic"), dict(ic50, assay_type="生化")), 1)
        self.assertEqual(record_tp(dict(ic50, variant="L858R/T790M"), dict(ic50, variant="T790M, L858R")), 1)
        self.assertEqual(record_tp(dict(ic50, variant="L858R/T790M"), dict(ic50, variant="L858R")), 0)
        pk = dict(endpoint="AUC", value=1200, unit="ng·h/mL", target="", dose=10, dose_unit="mg/kg", route="po")
        self.assertEqual(record_tp(pk, dict(pk, dose=10, dose_unit="mpk", route="oral")), 1)
        self.assertEqual(record_tp(pk, dict(pk, route="iv")), 0)
        self.assertEqual(record_tp(pk, dict(pk, dose=1)), 0)

    def test_compound_units_convert_by_dimension(self):
        cl = dict(endpoint="CL", target="")
        self.assertEqual(record_tp(dict(cl, value="25", unit="mL/min/kg"), dict(cl, value=1.5, unit="L/h/kg")), 1)
        self.assertEqual(record_tp(dict(cl, value="25", unit="mL/min/kg"), dict(cl, value=1.6, unit="L/h/kg")), 0)
        self.assertEqual(record_tp(dict(cl, value="1.5", unit="L/h/kg"), dict(cl, value=25, unit="mL·min⁻¹·kg⁻¹")), 1)
        auc = dict(endpoint="AUC0-inf", target="")
        self.assertEqual(record_tp(dict(auc, value="1.2", unit="µg·h/mL"), dict(auc, value=1200, unit="ng·h/mL")), 1)
        self.assertEqual(record_tp(dict(auc, value="1.2", unit="µg·h/mL"), dict(auc, value=1.2, unit="ng·h/mL")), 0)
        t = dict(endpoint="t1/2", target="")
        self.assertEqual(record_tp(dict(t, value="50", unit="min"), dict(t, value=0.83, unit="h")), 1)

    def test_decimal_multiplier_in_unit_or_header(self):
        papp = dict(endpoint="Papp A→B", target="")
        self.assertEqual(record_tp(dict(papp, value="12", unit="×10⁻⁶ cm/s"), dict(papp, value=1.2e-5, unit="cm/s")), 1)
        self.assertEqual(record_tp(dict(papp, value="12", unit="10-6 cm/s"), dict(papp, value="12 × 10⁻⁶", unit="cm/s")), 1)
        self.assertEqual(record_tp(dict(papp, endpoint="Papp A→B (10⁻⁶ cm/s)", value="12", unit=""),
                                   dict(papp, value=1.2e-5, unit="cm/s")), 1)
        self.assertEqual(record_tp(dict(papp, value="12", unit="cm/s"), dict(papp, value=1.2e-5, unit="cm/s")), 0)
        # transport direction stays part of the endpoint
        self.assertEqual(record_tp(dict(papp, endpoint="Papp (A→B)", value=1, unit="cm/s"),
                                   dict(papp, endpoint="Papp (B→A)", value=1, unit="cm/s")), 0)

    def test_percent_endpoints_and_logd_ph(self):
        self.assertEqual(record_tp(dict(endpoint="F", value=45, unit="%", target=""),
                                   dict(endpoint="F%", value=45, unit="", target="")), 1)
        self.assertEqual(record_tp(dict(endpoint="F", value=45, unit="%", target=""),
                                   dict(endpoint="oral bioavailability", value=45, unit="%", target="")), 1)
        logd = dict(value="2.1", unit="", target="")
        self.assertEqual(record_tp(dict(logd, endpoint="logD7.4"), dict(logd, endpoint="logD", ph=7.4)), 1)
        self.assertEqual(record_tp(dict(logd, endpoint="logD (pH 7.4)"), dict(logd, endpoint="logD7.4")), 1)
        self.assertEqual(record_tp(dict(logd, endpoint="logD7.4"), dict(logd, endpoint="logD", ph=6.5)), 0)
        self.assertEqual(record_tp(dict(logd, endpoint="logD7.4"), dict(logd, endpoint="logD")), 0)
        self.assertEqual(record_tp(dict(logd, endpoint="logD"), dict(logd, endpoint="logD7.4")), 1)


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

    def test_record_matching_is_maximal(self):
        # gold prints 50 and 50.4 nM; the model writes pIC50. 50.4 nM fits both rounding windows,
        # 49.8 nM only the window of "50": a greedy match would find one record, the best two.
        import math
        px = lambda nm: round(9 - math.log10(nm), 6)
        gold = doc([{"id": "1", "smiles": ASPIRIN}], [meas("1", 50), meas("1", 50.4)])
        pred = doc([{"id": "1", "smiles": ASPIRIN}],
                   [meas("1", px(50.4), unit="", endpoint="pIC50"), meas("1", px(49.8), unit="", endpoint="pIC50")])
        self.assertEqual(score.score_document(gold, pred)["record_tp"], 2)

    def test_unscorable_gold_records_excluded_from_both_sides(self):
        gold = doc([{"id": "1", "smiles": ASPIRIN}, {"id": "2", "smiles": "not-a-smiles"}],
                   [meas("1", 12.3), meas("2", 450.0)])
        pred = doc([{"id": "1", "smiles": ASPIRIN}, {"id": "2", "smiles": PARACETAMOL}],
                   [meas("1", 12.3), meas("2", 450.0)])
        s = score.aggregate([score.score_document(gold, pred)])
        self.assertEqual((s["record_precision"], s["record_recall"]), (1.0, 1.0))
        self.assertEqual(s["n_gold_rec_unscorable"], 1)

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

    def test_seed_variance_widens_the_interval(self):
        # same recipe, three seeds that disagree by several points
        base = [self._per_doc(n, 200, family_size=4) for n in (140, 150, 160)]
        new = [self._per_doc(n, 200, family_size=4) for n in (146, 154, 162)]
        pooled = score.paired_compare(score.pool_seeds(base), score.pool_seeds(new))
        hier = score.paired_compare(base, new)
        self.assertAlmostEqual(hier["delta"], pooled["delta"])
        self.assertGreater(hier["ci_hi"] - hier["ci_lo"], pooled["ci_hi"] - pooled["ci_lo"])
        self.assertEqual(hier["base_seed_scores"], [0.7, 0.75, 0.8])

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


    def _sliced(self, plan):
        """plan: [(slice, n_correct, n_total, tags)] -> per-doc rows."""
        rows = []
        for sl, n_correct, n_total, tags in plan:
            for i in range(n_total):
                gold = dict(doc([{"smiles": ASPIRIN}], doc_id=f"{sl}{i}"), slice=sl, tags=list(tags))
                rows.append(score.score_document(gold, doc([{"smiles": ASPIRIN if i < n_correct else PARACETAMOL}],
                                                           doc_id=f"{sl}{i}")))
        return rows

    def test_slice_weights(self):
        rows = self._sliced([("patent", 90, 100, ()), ("paper", 10, 20, ())])
        self.assertAlmostEqual(score.aggregate(rows)["score"], 100 / 120)  # micro-average
        w = score.parse_slice_weights("patent=1, paper=1")
        self.assertEqual(w, {"patent": 1.0, "paper": 1.0})
        self.assertAlmostEqual(score.aggregate(rows, slice_weights=w)["score"], (0.9 + 0.5) / 2)
        self.assertAlmostEqual(score.aggregate(rows, slice_weights={"paper": 1, "absent": 5})["score"], 0.5)
        self.assertIsNone(score.parse_slice_weights(""))

    def test_compare_reports_slices_tags_and_guardrails(self):
        base = self._sliced([("patent", 60, 100, ("stereo",)), ("paper", 60, 100, ())])
        new = self._sliced([("patent", 40, 100, ("stereo",)), ("paper", 90, 100, ())])
        r = score.paired_compare(base, new, slice_weights={"patent": 1, "paper": 1})
        self.assertAlmostEqual(r["delta"], 0.05)
        self.assertAlmostEqual(r["slices"]["patent"][0], -0.20)
        self.assertAlmostEqual(r["slices"]["paper"][0], 0.30)
        self.assertAlmostEqual(r["tags"]["stereo"][0], -0.20)
        self.assertGreater(r["tags"]["stereo"][1], 0.0)  # bootstrap SE
        self.assertEqual(set(r["guardrails"]), set(score.GUARDRAILS))
        self.assertEqual(r["guardrails"]["invalid_smiles_rate"], (0.0, 0.0))

    def test_smaller_alpha_gives_wider_interval(self):
        base, new = self._per_doc(60, 200), self._per_doc(70, 200)
        r90, r99 = score.paired_compare(base, new, alpha=0.10), score.paired_compare(base, new, alpha=0.01)
        self.assertLess(r99["ci_lo"], r90["ci_lo"])
        self.assertGreater(r99["ci_hi"], r90["ci_hi"])
        self.assertEqual(r90["p_not_better"], r99["p_not_better"])

if __name__ == "__main__":
    unittest.main()
