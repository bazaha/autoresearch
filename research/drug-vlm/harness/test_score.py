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

    def test_parent_is_not_the_largest_fragment(self):
        # pyrantel pamoate: the counterion is larger than the drug
        pyrantel, oxantel = "CN1CCCN=C1/C=C/c1cccs1", "CN1CCCN=C1/C=C/c1cccc(O)c1"
        pamoic = "OC(=O)c1cc2ccccc2c(Cc2c(O)c(C(=O)O)cc3ccccc23)c1O"
        key = score.structure_key
        self.assertEqual(key(pyrantel + "." + pamoic), key(pyrantel))
        self.assertNotEqual(key(oxantel + "." + pamoic), key(pyrantel + "." + pamoic))
        self.assertNotEqual(key(pamoic), key(pyrantel + "." + pamoic))
        self.assertEqual(key("CCN.Cc1ccc(S(=O)(=O)O)cc1"), key("CCN"))           # tosylate
        self.assertEqual(key("CC(=O)Oc1ccccc1C(=O)O[Mg]OC(=O)c1ccccc1OC(C)=O"), key(ASPIRIN))   # 2 equivalents
        self.assertNotEqual(key(ASPIRIN + ".CCCCCCCC"), key(ASPIRIN))           # an unlisted co-component stays

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
        self.assertEqual(score._norm_unit("µM"), score._norm_unit("umol/L") and "umolar")

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


def E(**k):
    return dict(dict(target="", unit=""), **k)


# (name, gold record(s), predicted record(s), expected record_tp). Found by red-teaming v1.6.
HEADER_CASES = [
    ("test concentration in brackets is a condition, not dropped",
     [E(endpoint="% inhibition (1 µM)", value=45, target="EGFR"), E(endpoint="% inhibition (10 µM)", value=80, target="EGFR")],
     [E(endpoint="% inhibition (1 µM)", value=80, target="EGFR"), E(endpoint="% inhibition (10 µM)", value=45, target="EGFR")], 0),
    ("bracketed concentration == conc field", E(endpoint="% inhibition (1 µM)", value=45, target="EGFR"),
     dict(E(endpoint="% inhibition", value=45, target="EGFR", unit="%"), conc=1, conc_unit="uM"), 1),
    ("unrecognized qualifiers (cell lines) keep columns apart",
     [E(endpoint="IC50 (HCT116)", value=1, unit="uM"), E(endpoint="IC50 (A549)", value=5, unit="uM")],
     [E(endpoint="IC50 (HCT116)", value=5, unit="uM"), E(endpoint="IC50 (A549)", value=1, unit="uM")], 0),
    ("WT and mutant columns stay apart",
     [E(endpoint="IC50 (WT)", value=1, unit="nM", target="EGFR"), E(endpoint="IC50 (L858R)", value=5, unit="nM", target="EGFR")],
     [E(endpoint="IC50 (WT)", value=5, unit="nM", target="EGFR"), E(endpoint="IC50 (L858R)", value=1, unit="nM", target="EGFR")], 0),
    ("comma list in a bracket", E(endpoint="CL (rat, iv)", value=25, unit="mL/min/kg"),
     E(endpoint="CL", value=25, unit="mL/min/kg", species="rat", route="iv"), 1),
    ("AUC intervals are distinct endpoints",
     [E(endpoint="AUC(0-t)", value=100, unit="ng·h/mL"), E(endpoint="AUC(0-∞)", value=120, unit="ng·h/mL")],
     [E(endpoint="AUC(0-t)", value=120, unit="ng·h/mL"), E(endpoint="AUC(0-∞)", value=100, unit="ng·h/mL")], 0),
    ("AUC interval spellings", E(endpoint="AUCinf", value=120, unit="ng·h/mL"), E(endpoint="AUC(0-∞)", value=120, unit="ng·h/mL"), 1),
    ("interval bracket is not the unit", E(endpoint="AUC(0-24h) (ng·h/mL)", value="1200"), E(endpoint="AUC0-24", value=1.2, unit="µg·h/mL"), 1),
    ("fullwidth text", E(endpoint="ＩＣ５０", value=5, unit="nM", target="EGFR"), E(endpoint="IC50", value=5, unit="nM", target="EGFR"), 1),
    ("fullwidth percent", E(endpoint="F", value=45, unit="％"), E(endpoint="F", value=45, unit="%"), 1),
    ("Japanese species", E(endpoint="CLint", value=12, unit="uL/min/mg", species="ヒト"),
     E(endpoint="CLint", value=12, unit="uL/min/mg", species="rat"), 0),
    ("Japanese endpoints", E(endpoint="阻害率", value=50, unit="%", target="EGFR"), E(endpoint="半減期", value=50, unit="%", target="EGFR"), 0),
    ("incubation time in a bracket", E(endpoint="IC50 (72 h) (µM)", value="1.2", target="EGFR"),
     E(endpoint="IC50", value=1200, unit="nM", target="EGFR", time="72 h"), 1),
    ("unit next to a statistic", E(endpoint="IC50 (nM, n=3)", value="12", target="EGFR"), E(endpoint="IC50", value=12, unit="nM", target="EGFR"), 1),
    ("parenthesized denominator", E(endpoint="CL", value=25, unit="mL/(min·kg)"), E(endpoint="CL", value=25, unit="mL/min/kg"), 1),
    ("target written before the endpoint", E(endpoint="EGFR IC50", value=5, unit="nM", target="EGFR"),
     E(endpoint="IC50", value=5, unit="nM", target="EGFR"), 1),
    ("assay type written before the endpoint", E(endpoint="Cellular IC50", value=5, unit="nM", target="EGFR"),
     E(endpoint="IC50", value=5, unit="nM", target="EGFR", assay_type="cell"), 1),
    ("unit after a slash", E(endpoint="IC50/nM", value=5, target="EGFR"), E(endpoint="IC50", value=5, unit="nM", target="EGFR"), 1),
    ("Chinese 值 suffix", E(endpoint="IC50值", value=5, unit="nM", target="EGFR"), E(endpoint="IC50", value=5, unit="nM", target="EGFR"), 1),
    ("± SD in the header", E(endpoint="IC50 ± SD (nM)", value="5", target="EGFR"), E(endpoint="IC50", value=5, unit="nM", target="EGFR"), 1),
    ("percent sign on a Chinese endpoint", E(endpoint="抑制率%", value=50, target="EGFR"),
     E(endpoint="% inhibition", value=50, unit="%", target="EGFR"), 1),
    ("matrix written before the endpoint", E(endpoint="HLM CLint", value=12, unit="uL/min/mg"),
     E(endpoint="CLint", value=12, unit="uL/min/mg", matrix="HLM"), 1),
    ("subscript pH", E(endpoint="logD₇.₄", value="2.1"), E(endpoint="logD7.4", value=2.1), 1),
    ("pH in a bracket keeps logD columns apart",
     [E(endpoint="LogD(7.4)", value="2.1"), E(endpoint="LogD(2.0)", value="0.5")],
     [E(endpoint="LogD(7.4)", value="0.5"), E(endpoint="LogD(2.0)", value="2.1")], 0),
    ("wild type spellings", E(endpoint="IC50", value=5, unit="nM", target="EGFR", variant="wild type"),
     E(endpoint="IC50", value=5, unit="nM", target="EGFR", variant="EGFR WT"), 1),
    ("a dash is not wild type", E(endpoint="IC50", value=5, unit="nM", target="EGFR", variant="WT"),
     E(endpoint="IC50", value=5, unit="nM", target="EGFR", variant="—"), 0),
    ("mutation written in the target", E(endpoint="IC50", value=5, unit="nM", target="EGFR", variant="L858R/T790M"),
     E(endpoint="IC50", value=5, unit="nM", target="EGFR T790M/L858R"), 1),
    ("strain and plural", E(endpoint="CL", value=25, unit="mL/min/kg", species="rat"),
     E(endpoint="CL", value=25, unit="mL/min/kg", species="SD rats"), 1),
    ("Chinese strain", E(endpoint="CL", value=25, unit="mL/min/kg", species="大鼠"),
     E(endpoint="CL", value=25, unit="mL/min/kg", species="SD大鼠"), 1),
    ("rhesus is not cynomolgus", E(endpoint="CL", value=25, unit="mL/min/kg", species="rhesus"),
     E(endpoint="CL", value=25, unit="mL/min/kg", species="cynomolgus"), 0),
    ("cell line suffix", E(endpoint="IC50", value=1, unit="uM", cell_line="A549"),
     E(endpoint="IC50", value=1, unit="uM", cell_line="A549 cells"), 1),
    ("assay type suffix", E(endpoint="IC50", value=5, unit="nM", target="EGFR", assay_type="biochemical"),
     E(endpoint="IC50", value=5, unit="nM", target="EGFR", assay_type="enzymatic assay"), 1),
    ("Chinese assay type", E(endpoint="IC50", value=5, unit="nM", target="EGFR", assay_type="cell"),
     E(endpoint="IC50", value=5, unit="nM", target="EGFR", assay_type="细胞水平"), 1),
    ("route spellings", E(endpoint="CL", value=25, unit="mL/min/kg", route="po"),
     E(endpoint="CL", value=25, unit="mL/min/kg", route="oral gavage"), 1),
    ("pH field with text", E(endpoint="logD", value=2.1, ph=7.4), E(endpoint="logD", value=2.1, ph="pH 7.4"), 1),
    ("dose as one string", E(endpoint="AUC", value=1200, unit="ng·h/mL", dose=10, dose_unit="mg/kg"),
     E(endpoint="AUC", value=1200, unit="ng·h/mL", dose="10 mg/kg"), 1),
    ("direction spelled 'to'", E(endpoint="Papp (A to B)", value=12, unit="10-6 cm/s"), E(endpoint="Papp A→B", value=12, unit="10-6 cm/s"), 1),
    ("direction next to the unit", E(endpoint="Papp (A→B, ×10⁻⁶ cm/s)", value=12), E(endpoint="Papp (B→A, ×10⁻⁶ cm/s)", value=12), 0),
    ("apical/basolateral", E(endpoint="Papp (AP→BL)", value=12, unit="cm/s"), E(endpoint="Papp (BL→AP)", value=12, unit="cm/s"), 0),
    ("Greek letter names", E(endpoint="IC50", value=5, unit="nM", target="GSK-3β"), E(endpoint="IC50", value=5, unit="nM", target="GSK3beta"), 1),
]


P = lambda **k: E(endpoint="Papp A→B", **k)
I = lambda **k: E(endpoint="IC50", target="EGFR", **k)

# Units and value strings (red team, v1.6).
UNIT_CASES = [
    ("multiplier with mantissa", P(value="12", unit="×10⁻⁶ cm/s"), P(value=12, unit="1×10⁻⁶ cm/s"), 1),
    ("multiplier in brackets", P(value="12", unit="×10⁻⁶ cm/s"), P(value=12, unit="(×10⁻⁶ cm/s)"), 1),
    ("en-dash exponent", P(value="12", unit="×10⁻⁶ cm/s"), P(value=12, unit="×10–6 cm/s"), 1),
    ("trailing multiplier", P(value="12", unit="×10⁻⁶ cm/s"), P(value=12, unit="cm/s ×10⁻⁶"), 1),
    ("a multiplier is never dropped", P(value="12", unit="cm/s"), P(value=12, unit="1×10⁻⁶ cm/s"), 0),
    ("dose in a bracket", E(endpoint="Cmax (10 mg/kg)", value=100, unit="ng/mL"), E(endpoint="Cmax (30 mg/kg)", value=100, unit="ng/mL"), 0),
    ("capital X exponent", I(value="1.2e-8", unit="M"), I(value="1.2 X 10-8", unit="M"), 1),
    ("spaced E exponent", I(value="1.2e-8", unit="M"), I(value="1.2 E-8", unit="M"), 1),
    ("HTML superscript", I(value="1.2e-8", unit="M"), I(value="1.2×10<sup>-8</sup>", unit="M"), 1),
    ("± with an exponent", I(value="1.2e-8", unit="M"), I(value="1.2 ± 0.3 × 10⁻⁸", unit="M"), 1),
    ("decimal comma is not dropped", I(value="5", unit="uM"), I(value="0,5", unit="uM"), 0),
    ("decimal comma", I(value="0.5", unit="uM"), I(value="0,5", unit="uM"), 1),
    ("tilde range", I(value=[10, 30], unit="nM"), I(value="10~30 nM"), 1),
    ("fullwidth tilde range is not a single value", I(value="10", unit="nM"), I(value="10～30 nM"), 0),
    ("range with repeated units", I(value=[10, 30], unit="uM"), I(value="10 µM–30 µM"), 1),
    ("fullwidth relations stay apart", I(value="＜10 nM"), I(value="＞10 µM"), 0),
    ("fullwidth relation field", I(value=10, unit="nM", relation="＜"), I(value=10, unit="nM", relation="＞"), 0),
    ("fullwidth less-than", I(value="<10", unit="nM"), I(value="＜10", unit="nM"), 1),
    ("≦", I(value="<10", unit="nM"), I(value="≦10", unit="nM"), 1),
    ("fullwidth unit", I(value=10, unit="nM"), I(value=10, unit="ｎＭ"), 1),
    ("en-dash minus", E(endpoint="logD", value=-0.5), E(endpoint="logD", value="–0.5"), 1),
    ("footnote marks do not make values collide", I(value="1.2†", unit="nM"), I(value="12*", unit="uM"), 0),
    ("footnote mark", I(value="12", unit="nM"), I(value="12*", unit="nM"), 1),
    ("superscript footnote letter", I(value="12", unit="nM"), I(value="12ᵃ", unit="nM"), 1),
    ("value with its range in brackets", I(value="12", unit="nM"), I(value="12 (10–15)", unit="nM"), 1),
    ("string concentrations stay apart", dict(I(value=50, unit="%"), endpoint="% inhibition", conc="10 µM"),
     dict(I(value=50, unit="%"), endpoint="% inhibition", conc="1 µM"), 0),
    ("string concentration", dict(I(value=50, unit="%"), endpoint="% inhibition", conc=1, conc_unit="uM"),
     dict(I(value=50, unit="%"), endpoint="% inhibition", conc="1 µM"), 1),
    ("unit qualifier word", E(endpoint="CLint", value=12, unit="µL/min/mg"), E(endpoint="CLint", value=12, unit="µL/min/mg protein"), 1),
    ("en-dash inverse exponents", E(endpoint="CL", value=1.5, unit="L/h/kg"), E(endpoint="CL", value=1.5, unit="L·h–1·kg–1"), 1),
    ("amount per volume is molar", E(endpoint="Cmax", value=1, unit="µM"), E(endpoint="Cmax", value=1, unit="nmol/mL"), 1),
    ("molar inside a compound unit", E(endpoint="AUC", value=5, unit="µM·h"), E(endpoint="AUC", value=5, unit="nmol·h/mL"), 1),
    ("Chinese unit words", I(value=10, unit="微摩尔/升"), I(value=10, unit="纳摩尔/升"), 0),
    ("Chinese = symbol", I(value=10, unit="μmol/L"), I(value=10, unit="微摩尔/升"), 1),
    ("Chinese time units", E(endpoint="t1/2", value=2, unit="小时"), E(endpoint="t1/2", value=2, unit="分钟"), 0),
    ("Chinese hours", E(endpoint="t1/2", value=2, unit="h"), E(endpoint="t1/2", value=2, unit="小时"), 1),
    ("a unit is never equal to no unit", E(endpoint="t1/2", value=2, unit="小时"), E(endpoint="t1/2", value=2, unit=""), 0),
    ("pH and unit in one bracket", E(endpoint="Solubility (pH 7.4, µM)", value="50"), E(endpoint="solubility", value=50, unit="uM", ph=7.4), 1),
    ("unit and matrix in one bracket", E(endpoint="CLint (µL/min/mg, HLM)", value="12"),
     E(endpoint="CLint", value=12, unit="uL/min/mg", matrix="HLM"), 1),
    ("unit after a slash in t1/2", E(endpoint="t1/2/h", value="2"), E(endpoint="t1/2", value=2, unit="h"), 1),
    ("fraction vs percent", E(endpoint="fu", value="0.05"), E(endpoint="fu", value=5, unit="%"), 1),
    ("percent vs fraction unit", E(endpoint="F", value=45, unit="%"), E(endpoint="F", value=0.45, unit="fraction"), 1),
]


class UnitParsingTests(unittest.TestCase):
    def test_unit_cases(self):
        for name, gold_m, pred_m, want in UNIT_CASES:
            gold = doc([{"id": "1", "smiles": ASPIRIN}], [dict(compound_id="1", **gold_m)])
            pred = doc([{"id": "1", "smiles": ASPIRIN}], [dict(compound_id="1", **pred_m)])
            with self.subTest(name):
                self.assertEqual(score.score_document(gold, pred)["record_tp"], want)

    def test_overflow_is_an_invalid_record_not_a_crash(self):
        for value, unit, endpoint in (("1e400", "nM", "IC50"), ("1.2 ×10^400", "nM", "IC50"), (-400, "", "pIC50")):
            self.assertIsNone(score.normalize_measurement(meas("1", value, unit=unit, endpoint=endpoint)))
        gold = doc([{"id": "1", "smiles": ASPIRIN}], [meas("1", 5)])
        pred = doc([{"id": "1", "smiles": ASPIRIN}], [meas("1", "1e400"), dict(meas("1", 5), dose="1", dose_unit="10^400 mg/kg")])
        self.assertEqual(score.score_document(gold, pred)["n_pred_rec"], 2)


class HeaderParsingTests(unittest.TestCase):
    def test_header_cases(self):
        for name, gold_m, pred_m, want in HEADER_CASES:
            g = gold_m if isinstance(gold_m, list) else [gold_m]
            p = pred_m if isinstance(pred_m, list) else [pred_m]
            gold = doc([{"id": "1", "smiles": ASPIRIN}], [dict(compound_id="1", **m) for m in g])
            pred = doc([{"id": "1", "smiles": ASPIRIN}], [dict(compound_id="1", **m) for m in p])
            with self.subTest(name):
                self.assertEqual(score.score_document(gold, pred)["record_tp"], want)


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


class ExploitTests(unittest.TestCase):
    """Found by red-teaming v1.6: ways to score higher without being more correct, and crashes."""

    def test_hedging_around_an_unscorable_gold_compound_does_not_pay(self):
        gold = doc([{"id": "1", "smiles": ASPIRIN}, {"id": "2", "smiles": "polymer-not-smiles"}], [meas("1", 10), meas("2", 20)])
        honest = doc([{"id": "1", "smiles": ASPIRIN}, {"id": "2", "smiles": PARACETAMOL}], [meas("1", 10), meas("2", 20)])
        hedge = doc([{"id": "1", "smiles": ASPIRIN}, {"id": "1b", "smiles": PARACETAMOL}], [meas("1", 10), meas("1", 99), meas("1b", 10)])
        h = score.aggregate([score.score_document(gold, honest)])
        g = score.aggregate([score.score_document(gold, hedge)])
        self.assertEqual((h["struct_precision"], h["record_precision"]), (1.0, 1.0))
        self.assertLess(g["struct_precision"], 1.0)
        self.assertLess(g["record_precision"], 1.0)

    def test_labels_do_not_collide(self):
        self.assertEqual(len({score._norm_label(x) for x in ("1-11", "11-1", "111")}), 3)
        self.assertEqual(score._norm_label("Excipient"), "excipient")
        self.assertEqual({score._norm_label(x) for x in ("Ex. 12a", "Compound 12a", "12 a", "cpd 12a")}, {"12a"})
        phenol, aniline = "Oc1ccccc1", "Nc1ccccc1"
        gold = doc([{"id": "1-11", "smiles": phenol}, {"id": "11-1", "smiles": aniline}], [meas("1-11", 10), meas("11-1", 20)])
        reordered = doc([{"id": "11-1", "smiles": aniline}, {"id": "1-11", "smiles": phenol}], [meas("1-11", 10), meas("11-1", 20)])
        swapped = doc([{"id": "1-11", "smiles": phenol}, {"id": "11-1", "smiles": aniline}], [meas("1-11", 20), meas("11-1", 10)])
        self.assertEqual(score.score_document(gold, reordered)["record_tp"], 2)
        self.assertEqual(score.score_document(gold, swapped)["record_tp"], 0)

    def test_malformed_predictions_never_crash(self):
        gold = doc([{"id": "1", "smiles": ASPIRIN}], [meas("1", 10)])
        many_groups = ("*F.*Cl.*Br.*I.*C.*O.*N.*CC.*OC.*C#N.*SC.*CF.c1ccccc1 |" +
                       ",".join(f"m:{i}:29.30.31.32.33.34" for i in (0, 2, 4, 6, 8, 10, 12, 14, 17, 20, 23, 26)) + "|")
        for pred in (doc([{"id": "1", "smiles": ASPIRIN}], [meas("1", [])]),
                     {"doc_id": "d1", "compounds": [{"id": "1", "smiles": ASPIRIN}], "measurements": [None]},
                     {"doc_id": "d1", "compounds": ["CCO"]},
                     {"doc_id": "d1", "compounds": [{"id": "1", "smiles": [ASPIRIN]}]},
                     doc([{"id": "1", "smiles": ASPIRIN}], [meas("1", 10 ** 400)]),
                     {"doc_id": "d1", "compounds": "CCO", "measurements": "x"},
                     doc([{"id": "1", "smiles": many_groups}]),
                     ["not a document"]):
            with self.subTest(pred=str(pred)[:60]):
                score.score_document(gold, pred)

    def test_runaway_and_pathological_strings_are_bounded(self):
        import time
        self.assertIsNone(score.structure_key("C" * 20000))   # used to segfault RDKit canonicalization
        t0 = time.time()
        for v in ("12" + "\n" * 2000 + "nM", "1" * 500 + "-" + "1" * 500 + "x", "1" * 140 + "-" + "1" * 140 + "x"):
            score.normalize_measurement(meas("1", v))
        self.assertLess(time.time() - t0, 1.0)

    def test_bad_prediction_lines_are_counted_not_fatal(self):
        import json, os, tempfile
        with tempfile.TemporaryDirectory() as tmp:
            gold_path, pred_path = os.path.join(tmp, "g.jsonl"), os.path.join(tmp, "p.jsonl")
            with open(gold_path, "w") as f:
                f.write(json.dumps(doc([{"smiles": ASPIRIN}], doc_id="d0")) + "\n")
            with open(pred_path, "wb") as f:
                f.write(json.dumps(doc([{"smiles": ASPIRIN}], doc_id="d0")).encode() + b"\n")
                f.write(json.dumps({"doc_id": ["d1"]}).encode() + b"\n")
                f.write(b"[" * 100000 + b"]" * 100000 + b"\n")
                f.write(b'{"doc_id":"d1","compounds":[{"smiles":"C\xff"}]}\n')
            per_doc, missing = score.score_files(gold_path, pred_path)
        self.assertEqual((missing, per_doc[0]["struct_tp"]), (0, 1))

    def test_label_primes_and_route_suffixes(self):
        self.assertNotEqual(score._norm_label("1′"), score._norm_label("1"))
        self.assertEqual(score._norm_label("1′"), score._norm_label("1'"))
        self.assertEqual(score._context({"route": "口服给药"}, {}), (("route", "po"),))
        self.assertEqual(score._context({"route": "oral administration"}, {}), (("route", "po"),))

    def test_markush_attachment_sets_do_not_collide(self):
        # floating groups attached to {1, 2} vs {12}: the old digit concatenation merged them
        key = score.structure_key
        a = key("*Cl.*F.c1ccc2ccccc2c1 |m:0:2.3,m:2:2|")
        b = key("*Cl.*F.c1ccc2ccccc2c1 |m:0:2,m:2:2.3|")
        self.assertIsNotNone(a)
        self.assertNotEqual(a, b)


class LabelListTests(unittest.TestCase):
    def test_a_list_of_labels_links_to_nothing(self):
        for raw in ("1;2", "12a；12b", "化合物1、2"):
            self.assertEqual(score._norm_label(raw), "", raw)
        self.assertEqual(score._norm_label("Example 1-2"), "1-2")
        for raw, label in (("cpd 3;", "3"), ("12a；", "12a"), ("化合物5、", "5")):   # a stray trailing separator
            self.assertEqual(score._norm_label(raw), label, raw)
        gold = doc([{"id": "1-2", "smiles": ASPIRIN}], [meas("1-2", 12.3)])
        pred = doc([{"id": "1-2", "smiles": ASPIRIN}], [meas("1;2", 12.3)])
        self.assertEqual(score.score_document(gold, pred)["record_tp"], 0)


class ConverterTests(unittest.TestCase):
    TABLE = """
table: {doc_id: "EXAMPLE-PAPER-1", page: 5, table_id: "T2"}
scaffold: {cxsmiles: "O=C(Nc1ccc([*:1])cc1)c1ccncc1"}
rgroups:
  - {compound_id: "12a", R1: "[H]"}
  - {compound_id: "12b", R1: "F"}
  - {compound_id: "12c", R1: "OMe"}
legend: {"A": "<10 nM"}
records:
  - {compound_id: "12a", endpoint: "IC50", target: "EGFR", value: "12.30", unit: "nM"}
  - {compound_id: "12a", endpoint: "IC50", target: "", cell_line: "HCT116", value: ">10", unit: "µM"}
  - {compound_id: "12c", endpoint: "IC50", target: "EGFR", value: "A", footnote: "*"}
---
table: {doc_id: "EXAMPLE-PAPER-1", page: 6, table_id: "T3"}
scaffold: {cxsmiles: "*c1ccc(*)cc1 |$_R1;;;;;X;;$|"}
rgroups:
  - {compound_id: "20", R1: "CF3", X: "Cl"}
  - {compound_id: "21", R1: "Xyz", X: "Cl"}
"""

    def _convert(self, text):
        try:
            import yaml
        except ImportError:
            self.skipTest("PyYAML not installed")
        import yaml_to_jsonl
        warnings = []
        docs = yaml_to_jsonl.convert(list(yaml.load_all(text, Loader=yaml.BaseLoader)), warn=warnings.append)
        return docs, warnings

    def test_abbreviations_are_pinned_and_attach_once(self):
        import yaml_to_jsonl
        from rdkit import Chem
        for label, smi in yaml_to_jsonl._ABBREVIATIONS.items():
            self.assertIsNotNone(Chem.MolFromSmiles(smi), label)
            self.assertEqual(smi.count("*"), 1, label)
        acid = lambda r: yaml_to_jsonl.enumerate_rows("[*:1]C(=O)O", [{"compound_id": "x", "R1": r}])[0][1]
        for r, expected in (("iBu", "CC(C)CC(=O)O"), ("sBu", "CCC(C)C(=O)O"), ("iPent", "CC(C)CCC(=O)O"),
                            ("tBu", "CC(C)(C)C(=O)O"), ("CH3", "CC(=O)O"), ("N(CH3)2", "CN(C)C(=O)O"),
                            ("4-F-Ph", "O=C(O)c1ccc(F)cc1"), ("3-OMe-phenyl", "COc1cccc(C(=O)O)c1"),
                            ("-OCH3", "COC(=O)O"), ("[*]C#N", "N#CC(=O)O")):
            self.assertEqual(score.structure_key(acid(r)), score.structure_key(expected), r)

    def test_converter_edge_cases(self):
        docs, warnings = self._convert("""
table: {doc_id: "D1", table_id: "T1"}
scaffold: {cxsmiles: null, smiles: "*c1ccccc1"}
rgroups:
  - {compound_id: "1", R: "OH"}
  - {compound_id: "2", R: "Cl"}
  - ["3", "Br"]
records:
  - {compound_id: "", endpoint: "IC50", value: "5", unit: "nM"}
---
table: {doc_id: "D1", table_id: "T2"}
scaffold: {cxsmiles: "not a smiles"}
rgroups:
  - {compound_id: "4", R1: "F"}
compounds:
  - {compound_id: "2", smiles: "C(C"}
---
table: {doc_id: "D1", table_id: "T3"}
scaffold: {cxsmiles: "[*:1]c1ccccc1"}
rgroups:
  - {compound_id: "4", R1: "F"}
""")
        smiles = {c["id"]: c["smiles"] for c in docs[0]["compounds"]}
        self.assertEqual(score.structure_key(smiles["1"]), score.structure_key("Oc1ccccc1"))   # bare * is R
        self.assertEqual(score.structure_key(smiles["2"]), score.structure_key("Clc1ccccc1"))  # invalid duplicate ignored
        self.assertEqual(score.structure_key(smiles["4"]), score.structure_key("Fc1ccccc1"))   # filled in by T3
        self.assertIsNone(docs[0]["measurements"][0]["compound_id"])                          # unlabeled: unlinked
        self.assertTrue(any("not mappings" in w for w in warnings))

    def test_invalid_yaml_file_is_skipped(self):
        import os, tempfile
        try:
            import yaml_to_jsonl
        except SystemExit:
            self.skipTest("PyYAML not installed")
        with tempfile.TemporaryDirectory() as tmp:
            good, bad = os.path.join(tmp, "good.yaml"), os.path.join(tmp, "bad.yaml")
            with open(good, "w") as f:
                f.write('table: {doc_id: "D2"}\nrecords:\n  - {compound_id: "1", endpoint: "IC50", value: "3"}\n')
            with open(bad, "w") as f:
                f.write('table: {doc_id: "D3"\nrecords: [\n')
            warnings = []
            tables = yaml_to_jsonl.load([bad, good], warn=warnings.append)
        self.assertEqual(len(tables), 1)
        self.assertTrue(any("bad.yaml" in w for w in warnings))

    def test_records_yaml_converts_and_scores(self):
        try:
            import yaml
        except ImportError:
            self.skipTest("PyYAML not installed")
        import yaml_to_jsonl
        warnings = []
        docs = yaml_to_jsonl.convert(list(yaml.load_all(self.TABLE, Loader=yaml.BaseLoader)), warn=warnings.append)
        self.assertEqual(len(docs), 1)
        smiles = {c["id"]: c["smiles"] for c in docs[0]["compounds"]}
        self.assertEqual(score.structure_key(smiles["12c"]), score.structure_key("COc1ccc(NC(=O)c2ccncc2)cc1"))
        self.assertEqual(score.structure_key(smiles["12a"]), score.structure_key("O=C(Nc1ccccc1)c1ccncc1"))
        self.assertEqual(score.structure_key(smiles["20"]), score.structure_key("FC(F)(F)c1ccc(Cl)cc1"))
        self.assertIsNone(smiles["21"])
        self.assertTrue(any("21" in w for w in warnings))
        self.assertEqual(docs[0]["measurements"][0]["value"], "12.30")   # printed form kept
        self.assertEqual(docs[0]["measurements"][2]["bin"], "<10 nM")
        gold = doc([{"id": "12a", "smiles": "O=C(Nc1ccccc1)c1ccncc1"},
                    {"id": "12c", "smiles": "COc1ccc(NC(=O)c2ccncc2)cc1"}],
                   [meas("12a", "12.30"), dict(meas("12a", ">10", unit="µM", target=""), cell_line="HCT116"),
                    dict(meas("12c", "A", unit=""))], doc_id="EXAMPLE-PAPER-1")
        s = score.aggregate([score.score_document(gold, docs[0])])
        self.assertEqual(s["record_f1"], 1.0)


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
        # crops: a missing answer costs what a wrong one costs, so precision drops too
        self.assertAlmostEqual(score.aggregate(per_doc)["struct_precision"], 1 / 3)


    def test_prediction_without_compound_id_is_a_format_error(self):
        import json, os, tempfile
        gold = doc([{"id": "12a", "smiles": ASPIRIN}], [meas("12a", 12.3)])
        bad = {"doc_id": "d1", "compounds": [{"id": "12a", "smiles": ASPIRIN}],
               "measurements": [{"compound": "12a", "value": 12.3, "unit": "nM", "endpoint": "IC50", "target": "EGFR"}]}
        with tempfile.TemporaryDirectory() as tmp:
            gold_path, pred_path = os.path.join(tmp, "gold.jsonl"), os.path.join(tmp, "pred.jsonl")
            with open(gold_path, "w") as f:
                f.write(json.dumps(gold) + "\n")
            with open(pred_path, "w") as f:
                f.write(json.dumps(bad) + "\n")
            with self.assertRaisesRegex(ValueError, "compound_id.*yaml_to_jsonl"):
                score.score_files(gold_path, pred_path)
            with open(pred_path, "w") as f:
                f.write(json.dumps(doc([{"id": "12a", "smiles": ASPIRIN}], [meas("12a", 12.3)])) + "\n")
            per_doc, _ = score.score_files(gold_path, pred_path)
        self.assertEqual(score.aggregate(per_doc)["record_f1"], 1.0)

    def test_other_records_yaml_key_mistakes_are_format_errors(self):
        import json, os, tempfile
        gold = doc([{"id": "12a", "smiles": ASPIRIN}], [meas("12a", 12.3)])
        bad_compounds = {"doc_id": "d1", "compounds": [{"compound_id": "12a", "smiles": ASPIRIN}],
                         "measurements": [meas("12a", 12.3)]}
        bad_records = {"doc_id": "d1", "compounds": [{"id": "12a", "smiles": ASPIRIN}], "records": [meas("12a", 12.3)]}
        with tempfile.TemporaryDirectory() as tmp:
            gold_path, pred_path = os.path.join(tmp, "gold.jsonl"), os.path.join(tmp, "pred.jsonl")
            with open(gold_path, "w") as f:
                f.write(json.dumps(gold) + "\n")
            for bad, pattern in ((bad_compounds, "compounds use 'id'"), (bad_records, "no measurements")):
                with open(pred_path, "w") as f:
                    f.write(json.dumps(bad) + "\n")
                with self.assertRaisesRegex(ValueError, pattern):
                    score.score_files(gold_path, pred_path)

    def test_schema_relation_cat_matches_a_printed_binned_value(self):
        gold = doc([{"id": "1", "smiles": ASPIRIN}], [dict(meas("1", "A", unit=""), relation="cat")])
        pred = doc([{"id": "1", "smiles": ASPIRIN}], [dict(meas("1", "A", unit=""), relation="")])
        self.assertEqual(score.score_document(gold, pred)["record_tp"], 1)

    def test_group_field_office_is_carried_and_unknown_fields_fail(self):
        rows = [score.score_document(dict(doc([{"smiles": ASPIRIN}], doc_id=f"d{i}"), office=off), doc([{"smiles": ASPIRIN}], doc_id=f"d{i}"))
                for i, off in enumerate(["CN", "CN", "US"])]
        self.assertEqual(sorted(d["slice"] for d in score.regroup(rows, "office")), ["CN", "CN", "US"])
        self.assertEqual(score.aggregate(score.regroup(rows, "office"))["n_docs"], 3)   # office is not a count
        with self.assertRaisesRegex(ValueError, "no gold document"):
            score.regroup(rows, "language")
        with self.assertRaisesRegex(ValueError, "not a groupable field"):
            score.regroup(rows, "jurisdiction")


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

    def test_abstaining_is_no_better_than_guessing(self):
        def crops(n_right, n_wrong, n_empty):
            rows = []
            for i in range(n_right + n_wrong + n_empty):
                gold = doc([{"smiles": ASPIRIN}], doc_id=f"d{i}")
                pred = doc([{"smiles": ASPIRIN}] if i < n_right else [{"smiles": PARACETAMOL}] if i < n_right + n_wrong else [],
                           doc_id=f"d{i}")
                rows.append(score.score_document(gold, pred))
            return score.aggregate(rows, scalar="struct_f1")["score"]
        self.assertAlmostEqual(crops(70, 30, 0), 0.70)
        self.assertAlmostEqual(crops(70, 0, 30), 0.70)
        self.assertLess(crops(60, 0, 40), crops(70, 30, 0))

    def test_rates_stay_in_unit_interval(self):
        gold = doc([{"id": "1", "smiles": ASPIRIN}, {"id": "2", "smiles": "not-a-smiles"}],
                   [meas("1", 12.3), meas("2", 450.0)])
        pred = doc([{"id": "1", "smiles": "C1CC"}, {"id": "2", "smiles": "C(C"}], [meas("1", 12.3), meas("2", 450.0)])
        s = score.aggregate([score.score_document(gold, pred)])
        self.assertEqual(s["invalid_smiles_rate"], 1.0)
        good = doc([{"id": "1", "smiles": ASPIRIN}, {"id": "2", "smiles": PARACETAMOL}], [meas("1", 12.3), meas("2", 450.0)])
        s = score.aggregate([score.score_document(gold, good)])
        self.assertLessEqual(s["record_f1_by_label"], 1.0)

    def test_families_spanning_slices_resample_together(self):
        rows = []
        for i in range(10):
            for dt in ("paper", "si"):
                gold = dict(doc([{"smiles": ASPIRIN}], doc_id=f"a{i}-{dt}"), family_id=f"a{i}", doc_type=dt)
                rows.append(score.score_document(gold, doc([{"smiles": ASPIRIN if i % 2 else PARACETAMOL}], doc_id=f"a{i}-{dt}")))
        by_type = score.regroup(rows, "doc_type")
        self.assertEqual({d["slice"] for d in by_type}, {"paper", "si"})
        r = score.paired_compare(by_type, by_type, slice_weights={"paper": 1, "si": 1})
        self.assertEqual(r["n_clusters"], 10)

    def test_slice_weights_must_cover_every_slice(self):
        rows = self._sliced([("mrbw", 5, 10, ()), ("MRBW", 0, 10, ())])
        with self.assertRaises(ValueError):
            score.check_slice_weights(rows, {"mrbw": 1.0})
        used, absent = score.check_slice_weights(rows, {"mrbw": 1.0, "inhouse": 1.0}, allow_unweighted=True)
        self.assertEqual((used, absent), ({"mrbw": 1.0}, ["inhouse"]))

    def test_compare_refuses_runs_scored_on_different_gold(self):
        a, b = self._per_doc(60, 100), self._per_doc(60, 120)
        with self.assertRaises(ValueError):
            score.paired_compare(a, b)
        self.assertEqual(score.paired_compare(a, b, require_same_docs=False)["n_shared"], 100)
        tagged = lambda rows, h: [dict(d, gold=h) for d in rows]
        with self.assertRaises(ValueError):
            score.paired_compare(tagged(a, "aaa"), tagged(a, "bbb"))

    def test_one_seed_side_borrows_seed_variance(self):
        # Three stored seeds that disagree a lot (accuracy 0.5 / 0.7 / 0.9) against one new seed:
        # the interval must reflect that seed spread, not only the document resampling.
        base = [self._per_doc(n, 200) for n in (100, 140, 180)]
        new = self._per_doc(140, 200)
        screen = score.paired_compare(base, new)
        self.assertGreater(screen["ci_hi"] - screen["ci_lo"], 0.3)   # ~2 x 1.645 x 0.2
        same = score.paired_compare(base, base)
        self.assertEqual(same["delta"], 0.0)
        # Identical stored seeds (no seed effect): nothing is borrowed, the result equals 1 vs 1.
        one = score.paired_compare(base[0], new)
        triple = score.paired_compare([base[0]] * 3, new)
        self.assertGreater(one["ci_hi"] - one["ci_lo"], 0.05)
        self.assertAlmostEqual(one["ci_lo"], triple["ci_lo"], places=9)
        self.assertAlmostEqual(one["ci_hi"], triple["ci_hi"], places=9)

    def test_null_simulation_runs_against_this_scorer(self):
        import numpy as np
        import null_sim
        saved = score.BOOTSTRAP_SAMPLES
        try:
            pooled, hier, seeds = null_sim.trial(np.random.default_rng(0), 0.03)
            promising, _ = null_sim.screen_trial(np.random.default_rng(0), 0.03)
        finally:
            score.BOOTSTRAP_SAMPLES = saved
        self.assertEqual(len(seeds), null_sim.SEEDS)

if __name__ == "__main__":
    unittest.main()
