# Design B: a data-centric autoresearch loop for a drug-discovery extraction VLM

**Assumptions (2026-10-15).** One node with 8×H100-80GB, 96 CPU cores and 4 TB NVMe; a 1×H100 fallback is in §5. Commercial use is likely, so the inner loop trains only on a commercial-safe license track. Citations: AR = autoresearch-ecosystem, OCSR = ocsr-sota, DOC = doc-chem-ie, VLM = vlm-finetune-practice, DATA = data-sources, MET = metrics-eval-design. [U] marks findings the fact-checker could not verify.

---

## 1. Thesis

Freeze the base VLM, the optimizer and the evaluator. Make the training data the agent's editable surface: which molecules, drawn how, degraded how, labeled in which representation, mixed and ordered how.

The largest documented gains in this field are data effects. Augmentation plus a curriculum took MolParser from 40.1 to 76.9 on WildMol-10K (DATA F6). Superatom-faithful labels lifted MolScribe and MolNexTR from under 20% to over 70% on abbreviated images (OCSR F29). MinerU2.5-Pro gained +2.71 points with a frozen architecture (DOC F3). A data-selection agent found recipes that transferred from 125M to 897M models (AR F39).

Agents that edit model code mostly retune hyperparameters (AR F28), and operators matter more than search policy (AR F16). Data rankings at small scale pick the large-scale winner about 80% of the time (AR F41), while LR optima drift with horizon (AR F42). That makes a recipe a better unit for keep/discard. A kept recipe is also a durable asset: a frozen shard spec that survives a change of base model.

The cost is a new attack surface. A data generator is an ideal channel for smuggling eval content into training. Most of this design closes that channel mechanically while still giving the agent enough error signal to steer.

## 2. Staged scope

**Stage A, crop-level OCSR (weeks 1–3).** Input: one cropped molecule or Markush depiction; output: one structure, covering stereo, abbreviations, charges, large molecules and Markush CXSMILES. Structure goes first because it is the join key of every record, and BioMiner's end-to-end losses sit there: structure recall 0.58 vs value recall 0.85, and removing the structure module drops triplet F1 from 0.323 to 0.011 (DOC F9, DATA F23). Crops also keep evaluation to about a minute.

**Exit A.** All three conditions must hold, plus human sign-off:
1. The incumbent beats the warm start W0 on the hidden confirm tier: paired 95% CI lower bound above 0, and at least +5 pp on the real-literature slice.
2. Rung-1 vs rung-3 Spearman ρ ≥ 0.6 over at least 15 recipe pairs.
3. Either the targets are met or the loop has plateaued:
   - Targets: the 8B transfer reaches ≥70% SMILES exact match on our MolRecBench-Wild (MRBW) confirm split (GPT-5.6-Sol scores 74.9 and MolScribe 62.3, OCSR F10), and ≥50% CXSMILES exact match on IP5-M confirm (MarkushGlyph-2B scores 58.2, OCSR F26).
   - Plateau: 3 nights in a row with no confirmed gain.

**Stage B, region-level SAR records (weeks 4–6).** Input: a region (SAR table with scaffold drawing and footnotes, or a paragraph with values) cut by a frozen layout model (PaddleOCR-VL-0.9B, Apache-2.0, VLM F9); detection is not optimized, being about 0.99 mAP50 on born-digital PDFs (DOC F33). Output: compounds plus measurement records. New levers: synthetic SAR tables and real regions aligned to BindingDB/ChEMBL.

**Exit B.** Confirm record F1 is at least W0-B + 15 pp, record precision is at least 0.85, no Stage-A slice regresses by more than 1 pp, and ρ ≥ 0.6 again.

**Stage C, whole documents (week 7 onward).** Adds "Example 212" coreference across pages, Markush claims vs exemplified compounds, and provenance. It starts only after a program.md rewrite and new page-level gold.

Scope widens only when a human commits a new `STAGE` constant in prepare.py, which brings new tiers and pools. The agent cannot widen it.

## 3. File layout and the fixed/editable contract

```
research/drug-vlm/
  prepare.py      FIXED   STAGE, TOKEN_BUDGET, MAX_PIXELS, SEEDS; builds pools, eval tiers, blocklists
  generators/     FIXED   RDKit/Indigo/CDK renderers, augment ops, Markush & SAR-table generators,
                          label serializers, abbreviation dictionary (versioned gen_vN)
  recipe.py       AGENT   the data recipe: primary lever
  train.py        AGENT*  fixed trainer; only the "# --- KNOBS ---" block is editable
  program.md      HUMAN   priors (§7), scoring contract, simplicity rule, data policy
  runner.py       FIXED   gate -> datagen -> train -> eval -> accept -> ledger
  harness/        FIXED   score.py, evaluate.py (owned inference + NLL), decode_contract.py,
                          accept.py, gate_code.py, gate_data.py, receipts.py, tests
  data_engine/    HUMAN-GATED  harvest.py, vote.py, select.py, annotate/, program_engine.md
  lineage/        RUNNER-WRITTEN  ledger.jsonl (append-only), results.tsv, failed_ideas.tsv
```

**Why a second editable file, not train.py:** the recipe runs in a different sandbox (CPU only, different mounts); its output is content-hashed and cached, so a KNOBS change reuses the same data; the gate audits its output, not only its code; and each diff stays one kind of hypothesis.

**KNOBS in train.py.** The editable knobs are LR (5e-5 to 3e-4), LoRA rank (16 to 128), targets (attention, or attention plus MLP gate/up) and warmup. At most 20% of trials may touch them. Data conclusions can flip under HP changes (AR F42), so data is compared at fixed HPs.

```python
# recipe.py (sketch)
TARGET_FORMAT = "smiles_superatom"   # smiles | smiles_superatom | cxsmiles | graph_json | gtr_traversal
def build(ctx):                       # train-split pools, generators, rng, budget, aggregate error feed
    stereo = ctx.pools.pubchem.filter(n_stereo__gte=1)
    mix = {stereo: .25, ctx.pools.chembl: .35, ctx.pools.uspto_cwu["remap_star"]: .30, ctx.pools.replay: .10}
    for i in range(ctx.budget.max_samples):
        t = i / ctx.budget.max_samples                        # curriculum progress
        yield ctx.gen.depict(ctx.sample(mix), engine=ctx.rng.choice(["rdkit", "indigo", "cdk"]),
                             abbreviate=0.5, aug=ctx.gen.aug.ladder(0.2 + 0.8 * t))
```

**Sample specs, not pixels.** The recipe yields specs: pool item ID, generator, parameters, seed. A fixed `materialize` step renders them. Custom augmentations are allowed as `@ctx.custom_aug` functions (numpy, PIL or cv2 only). Promoting one into `generators/` takes a PR reviewed by a human.

**Enforcement is mechanical, not instructional:**
1. **Mounts and UIDs.** The agent container holds the repo. prepare.py, generators/, harness/ and runner.py are root-owned and read-only. There is no `/eval` mount, no network and no GPU. The agent only calls `uv run runner.py submit <sha>`, and the runner runs everything else under another UID.
2. **`gate_code.py`** rejects a commit that touches paths other than `recipe.py`, `train.py` or `notes/`; changes train.py's AST outside KNOBS (pinned hash); sets a KNOB out of range; imports anything outside numpy/PIL/cv2/rdkit/math/random or uses `open`/`socket`/`subprocess`/`eval`/`exec` in recipe.py; or contains a string literal over 8 characters matching a SMILES, InChIKey, DOI or patent-number regex.
3. **Datagen sandbox:** CPU only, read-only pools, no network, 16 cores, 8-minute cap; its manifest goes to `gate_data.py` (§8).
4. **Train sandbox:** mounts `/shards/<run>` read-only plus sha256-checked base weights; writes only `adapter.safetensors` and an enum-only `infer.json`; the fixed trainer counts tokens and steps.
5. **Eval sandbox:** separate UID, the only place gold is mounted; harness-owned vLLM greedy decoding with fixed max_tokens and `trust_remote_code=False`; writes a signed `score.json`. The ledger never parses stdout (MET F29).
6. **Pre-receive hook** on the runner's bare repo re-runs gate_code.

## 4. The metric

**Primary scalar.** `harness/accept.py` computes it from score.py's per-document output.
- **Stage A:** S_A is the slice-weighted structure F1: real-literature crops 0.45, hard (stereo plus MOSAIC subset C) 0.20, Markush 0.20, patent crops 0.15. This is close to the 0.5/0.2/0.2/0.1 split in the OCSR implications. Matching uses score.py's stereo-aware standard InChIKey of the FragmentParent+Uncharger parent, and a CXSMILES key for dummy-atom structures. On single-structure crops, F1 is the same as exact match.
- **Stage B:** S_B is the macro average over patents, papers and SI of score.py's composite (0.5·struct_F1 + 0.5·record_F1). S_A stays as a guardrail.

**Representation independence (the val_bpb analogue).** `decode_contract.py` turns every allowed `TARGET_FORMAT` into an RDKit mol: superatoms are expanded with a frozen dictionary (an unknown abbreviation is wrong, never `*`, MET F2), `graph_json` is assembled directly, and `gtr_traversal` is scored on its final SMILES. Formats can change freely; the score still means "same molecule".

**Continuous companion.** The teacher-forced NLL of the gold label, in nats per structure, on dev-fast. It is comparable only within one format. It ranks augmentation and mixture changes, breaks ties, and is tracked against S (MET F26). It never decides a change of format.

**Guardrails (each vetoes a keep):** invalid-structure rate +0.5 pp; structure or record precision −1 pp; any slice −max(2 pp, 2·SE_slice); orphan-record rate +1 pp; parse failures >0.5%; 300-item general probe (DocVQA/OCRBench) −2 points (VLM implications); canary accuracy diverging from dev by >5 pp; stereo vs no-stereo gap widening >1 pp; evaluation over 6 min.

**Eval tiers.** prepare.py builds them once. Splits are by patent family or DOI. Dedup uses InChIKey block 1 and pHash, plus the series Murcko scaffold for record tiers (MET implications).

| Tier | Agent sees | Stage A | Stage B |
|---|---|---|---|
| dev-open | items, gold, predictions | 600 crops (300 MRBW, 300 in-house SI) | 20 docs |
| dev-fast | aggregates rounded to 0.1 pp; per-tag error rates (n≥30) | 3,000 crops: 1,000 MRBW revised (SMILES-evaluable, stratified by subset A/B/C), 331 ACS, 300 corrected JPO/CLEF, 300 IP5-M, 500 BioVista-OCSR (chiral/Markush), 569 in-house | ≥100 docs / ≥3,000 records (60 in-house, 20 ChemX, 20 BioVista-val) |
| confirm | pass/fail only | 3,000 disjoint crops (1,000 MRBW, 1,200 BioVista-OCSR, 400 corrected CLEF/JPO, 400 in-house) plus the rest of IP5-M (578), M2S (103), USPTO-Markush (74) | ≥300 docs / ≥5,000 records, 40% in-house |
| sealed | nothing; human-run | 1,000 in-house crops from 2026 documents, from journals and offices (EPO/CNIPA) not used elsewhere | 100 double-annotated 2026 docs |

**Licenses.** All public sets are eval-only and never trained on; legal signs off on internal use. MRBW declares no license (OCSR F11), Keylab sets are mixed (OCSR F38), the Markush sets are CC BY 4.0 (DATA F14; the card has no tag), BioVista and ChemX are MIT. Only in-house sets are unencumbered.

**Sizing.** With 3,000 paired crops and 5% discordance, SE_diff ≈ √(0.05/3000) = 0.41 pp, so the decision margin (2·SE) is 0.8 pp (MET F22). Records cluster within documents, with a design effect of about 2.9 (MET F22), so record tiers need at least 300 documents.

**Gaps in score.py.** These get fixed as v1.1 in week 1, with golden tests:
1. **No slices.** Micro-F1 lets one 500-compound patent dominate; add `slice`, `family_id`, `tags`.
2. **Bootstrap unit.** It resamples `doc_id`, but pages of one patent correlate; cluster on `family_id`.
3. **Asymmetric exclusion.** Unscorable gold leaves the denominator, but predictions on it still count as false positives; exclude both and report them as a track.
4. **Markush.** Matching is canonical-CXSMILES string equality only: no substituent-table F1, no m/Sg check, and the R-label spelling (`[1*]` vs `R1`) depends on the format. Add table F1 and deterministic parsed-graph equality, not MCS with a timeout (MET F13).
5. **Stereo policy.** It is implicit: defined vs undefined stereo counts as a mismatch, and enhanced stereo is ignored. Declare an adaptive rule (OCSR F13). `assert` RDKit 2026.09.1 instead of printing the version, and add 200 golden pairs (MET F7).
6. **Measurement normalization.** `"IC50 (µM)"` normalizes to `"ic50m"` ≠ `"ic50"`; "EGFR kinase" and "ErbB1" both miss "EGFR"; `mL·min−1·kg−1` → `mlmin1kg1` ≠ `mlminkg`; %inhibition has no concentration field; ranges, n.d. and ±SD are unsupported; pX converts only when the unit is empty. Stage B needs closed vocabularies and a unit table.
7. **Greedy matching.** Records are matched greedily. Use Hungarian matching.
8. **No enforcement.** There are no hashes or receipts, and guardrails are reported but never enforced. Decisions belong in accept.py.

## 5. Budget per experiment

**Fixed per rung-1 trial:** `TOKEN_BUDGET` = 7.5M training tokens (vision plus text, counted by the fixed trainer); `MAX_PIXELS` = 512², i.e. 256 vision tokens in Qwen3-VL (VLM F32); the seed paired with the incumbent's.

A token budget stops throughput tricks from buying samples (AR F5, F36), and verbose formats pay for their own tokens. A crop is about 256 + 60 + 110 = 430 tokens, so the budget covers about 17k crops. A Stage-B region is about 1,900 tokens, so the same budget covers about 3.9k regions and the trial time stays flat.

**Wall clock.**
- **Training.** VLM F33 [U] estimates 10–45 crops/s for a 2B model on an H100. We use 20/s, about 8.6k tokens/s, or about 10% MFU at 6·N FLOPs per token. That gives 17k / 20 ≈ **14.5 min**.
- **Fixed costs.** Model load takes 1.5 min. vLLM start plus LoRA load takes 1 min. Decoding 3,000 crops at ≥50/s takes 1 min (VLM F34 [U]). The NLL pass takes 0.5 min and scoring 0.3 min.
- **Total.** About **19 min**, or **3.2 trials per hour per GPU**. A trial is killed at 27 min (1.4× nominal).

**Pre-generation.** Datagen runs on CPU, pipelined: trial k+1 materializes while trial k trains. Assuming ≥30 augmented crops/s per core (calibrate on day 1), 17k crops take about 36 s on 16 cores; Stage-B HTML tables at about 2 pages/s per core take about 2 min. A content-addressed cache keyed on (generator version, item, parameters, seed) uses about 0.5 GB per trial with LRU eviction. Datagen costs no GPU time, and the 8-minute cap guarantees it never stalls a GPU.

**One night (12 h) on 8×H100:**

| GPUs | Role | Arithmetic | Output |
|---|---|---|---|
| 0–5 | workers | 72 GPU-h: 80% → 57.6 / 0.32 h; 20% → 14.4 / (2 × 0.32 h) | **≈180 rung-1 trials, ≈22 rung-2 confirmations** |
| 6 | rung 3 | top-3 recipes at 4× tokens (30M ≈ 70k crops ≈ 1 h each) + confirm eval | 3 promotions |
| 7 | 8B transfer | incumbent on Qwen3-VL-8B LoRA, 30M tokens at about 5 crops/s [U] ≈ 3.9 h | 2 runs; daytime: data-engine teachers |

**Rungs:** 0 = gates, data card and a 60-s smoke test (OOM, NaN, throughput); 1 = 7.5M tokens on dev-fast; 2 = two more seeds; 3 = 4× tokens on confirm, plus 8B weekly.

Spearman ρ between rung 1 and rung 3 is computed over a rolling window of 15 pairs. If ρ < 0.6, rung-1 tokens are doubled (AR implications).

**Warm start.** W0 is Qwen3-VL-2B plus a LoRA trained on 300k commercial-safe crops (about 4.2 h), merged. Each trial trains a fresh LoRA on W0, measuring a recipe's marginal value, the regime the shipped model will be in. Weekly, the human re-cuts W_{n+1} with the frontier recipe at 300k samples: "advancing the branch" in data space.

**1-GPU fallback.** About 38 rung-1 trials per night with rung 2 inline. Rung 3 and the 8B transfer run on weekends.

## 6. Keep/discard under noise

In week 1, train the W0 recipe with 5 seeds to measure SE_seed and SE_item. The best of 180 null trials sits about 2.7 SE above zero (MET F24). An "any improvement" rule would therefore keep about a dozen false winners every night.

- **PROMISING (rung 1).** `python harness/score.py compare --base inc.jsonl --new cand.jsonl` (paired bootstrap, clustered by family) shows Δ ≥ max(0.8 pp, 2·SE_diff) and p_not_better ≤ 0.05, with no guardrail veto and NLL no worse than 2σ when the format is unchanged.
- **ESTABLISHED (rung 2).** Train two more seeds. The candidate must beat the incumbent's three stored seeds pairwise, 3 out of 3; a null change passes with probability 1/8 before any margin (AR F8). The pooled CI on Δ must have its lower bound above 0. Only then does the branch advance and the shard spec freeze.
- **Simplicity.** Discard a recipe change that adds more than 40 lines for under 1.0 pp. Keep a deletion that leaves the score equal.
- **PROMOTED (rung 3, nightly).** The top-3 ESTABLISHED nodes are scored on the hidden confirm tier under a rule locked in advance: paired, stratified bootstrap CI lower bound above 0, and no slice down by more than 2 pp. In AR F9 a dev gain halved under exactly this test. Selection uses the top-k plus confirmation, never the dev-best alone (AR F16). Keeps that fail are reverted on the frontier and marked "dev-only" in lineage.
- **Confirm budget.** At most 3 queries per night and 60 per month. After that, 30% of confirm items rotate in from a reserve set, in the spirit of Thresholdout (MET F25 [U]).
- **Sealed test.** Run by a human only, at stage exits and in week 6.
- **Ladder feedback.** Scores are rounded to 0.1 pp and slice deltas to 0.5 pp.

## 7. The agent's search space (prioritized)

| # | Lever in recipe.py | Expected impact | Evidence |
|---|---|---|---|
| 1 | Abbreviation-faithful labels (`smiles_superatom`); dictionary abbreviation substitution in renders | Very high on abbreviation slice | OCSR F29, F9; DATA F28 |
| 2 | Augmentation ladder with weak→strong curriculum; short SMILES first | High (40.1→69.5→76.9) | DATA F6 |
| 3 | Engine/style diversity (RDKit/Indigo/CDK, MaxMin style selection) | High OOD (+17.6 pp ChemDraw) | DATA F6, F31 |
| 4 | Real USPTO-CWU crop fraction, 10–60% | High on patents | DATA F3; DATA F7 [U] |
| 5 | USPTO label variant: raw / `*`→visible label / CDX-preferred / ensemble-filtered | Medium-high | DATA F2; OCSR F8 |
| 6 | Stereo oversampling (Stereo-200k ×2) + wedge↔wavy/bold swaps | High on chiral slice (weakest, 0.35–0.65) | OCSR F30, F26; DOC F34 |
| 7 | Decorative-wedge negatives (achiral molecules drawn with wedges) | Medium | OCSR F30 |
| 8 | Context distractors: neighbors, arrows, table borders, foreign script, colored backgrounds, truncation | Medium-high on MRBW subset B | OCSR F31, F11 |
| 9 | Scan/patent degradation: G4 binarization, 256-px downsampling, JPEG, blur | Medium on JPO/CLEF | OCSR F31 |
| 10 | Large-molecule (>70 atoms) oversampling, aspect-preserving padding | Medium | OCSR F1, F31 |
| 11 | Output format: SMILES vs `gtr_traversal` (+2.3) vs `graph_json` | Medium, token-cost aware | OCSR F9, F27, F28 |
| 12 | Randomized vs canonical SMILES labels | Unknown (own hypothesis) | OCSR F4 |
| 13 | Markush generator: R/m/Sg rates, template text, 10% paraphrase, 5% plain OCSR | High on IP5-M | DATA F13; OCSR F26 |
| 14 | Share of real USPTO-MOL-M (54,785) | Medium-high | DATA F14; OCSR F25 |
| 15 | Hard mining: up-weight mid-band incumbent NLL, skip noisy top | Medium | DOC F32; DATA F5 |
| 16 | MOSAIC-tag-targeted synthesis from aggregate error feed | Medium-high | DATA F27 |
| 17 | General replay share 0–30% | Forgetting guard | AR F9; VLM F25 |
| 18 | Charges, counterion salts (parent label), isotopes, hydrides | Small-medium | MET F6; OCSR F31 |
| 19 | Unique samples vs repeats within the budget | Medium | AR F39 |
| 20 | Weight of the consensus pseudo-label pool | High once pools exist | DATA F9; OCSR F6 |
| 21 | *(B)* Synthetic SAR tables from real-layout perturbation with ChEMBL/BindingDB rows (nM/µM/pX, qualifiers, ranges, ±SD, footnotes) | High; only scalable exact-label source | DATA F10, F38; DOC F30 |
| 22 | *(B)* BindingDB-aligned USPTO example tables as weak real records | High on patents | DATA F20 [U]; DOC F26 [U] |
| 23 | *(B)* Serialization: YAML vs JSON-lines, flat records | Medium (fewer loops) | VLM F13, F14 |
| 24 | *(B)* Scaffold + R-table renders labeled with RDKit-enumerated compounds | High on R-group tables | DOC F23 |

program.md tells the agent where to start (items 1, 2, 6, 4) and lists known negatives: captioners trained on synthetic data alone generalize poorly to real documents (OCSR F4), and generic synthetic layouts hurt (DATA F10).

## 8. Data strategy

Every pool item carries a `license_track`. Datagen refuses any non-C item in a commercial run.

**Track C (the only track the inner loop uses):** PubChem and ChEMBL SMILES as render sources (ChEMBL CC BY-SA attribution applies, DATA F18 [U]); USPTO complex-work-unit TIF+MOL/CDX pairs 2001–2024, minus benchmark patents and eval families, checking 37 CFR 1.71 notices (DATA F1 [U], F2, F37 [U]); MolScribe HF data (MIT tag, DATA F3); Stereo-200k (Apache); MarkushGrapher-1/2 data incl. USPTO-MOL-M (CC BY 4.0, DATA F13, F14); PMC OA commercial subset and CC BY ChemRxiv (DATA F33, F34 [U]); in-house documents; data-engine pools derived from these.

**Track R (a research leaderboard only, never merged into Track C models):** MolParser-7M, MolGallery, MolParser-Mobile and MolDetv2 (CC BY-NC-SA; DATA F4, F9), plus the COMO weights (NC).

**Synthetic generation** uses fixed generators: RDKit (BSD), Indigo (Apache) and CDK (LGPL, unmodified, separate process) with about 40 style parameters and the MolScribe/MolGrapher augmentation ops (DATA F6, F12, F31); Markush via CXSMILES augmentation, CDK rendering and template substituent text with 10% paraphrase (DATA F13); SAR tables only by perturbing real layouts, since generic synthetic pages hurt Uni-Parser (DATA F10).

**Label fidelity** (fixed rules; the recipe picks among precomputed variants): labels describe what is drawn, so superatoms stay superatoms and `*` appears only for a drawn wildcard; USPTO `*` atoms are remapped to the visible labels; CDX wins over MOL on disagreement; stereo is labeled only where a wedge is drawn.

**Dedup and leakage.**
- **prepare.py** strips from every pool: eval families and DOIs, eval InChIKey block-1 keys, pHash near-duplicates of eval images, and the source patents of classic benchmarks. USPTO, CLEF and Staker share one image pool with USPTO training sets (DATA F30).
- **`gate_data.py`** re-checks every materialized sample, because generators can recreate eval molecules (for example, R-group enumeration on a pool scaffold). Record tiers also block the series scaffold.
- **The blocklist** exists only inside the gate container. Hits are silently replaced by resampled pool items. The agent sees only a rejection bucket per trial: 0, <0.5% or ≥0.5%. That stops the gate from working as a membership oracle. A rate of ≥0.5% alerts a human.

**Fixed:** pools, tiers, blocklists, generator library, label variants, MAX_PIXELS, token budget, decoding. **Agent-tunable:** pool and variant choice, weights, filters, generator parameters, custom augmentations, curriculum, `TARGET_FORMAT`, hard-mining rules, replay share.

**Data engine.** A separate loop, gated by a human, run weekly, and governed by `data_engine/program_engine.md`:
1. **Harvest.** Unlabeled Track-C crops (USPTO pages, OA-commercial and CC BY figures, in-house SI), detected by our own detector because MolDetv2 is NC (DATA F11), bootstrapped from Docling's MIT figure classifier (DOC F3). Born-digital PDFs are vector-parsed ChemScraper-style for near-exact labels (OCSR ADDED).
2. **Vote.** Voters: the promoted model, MolScribe (MIT), OCSRGlyph/MarkushGlyph and MarkushGrapher-2 (Apache), DECIMER (MIT), and OPSIN where an IUPAC name exists. Accept when ≥3 architecturally different voters agree on the full InChIKey, behind a substance filter against wildcard-only agreement (MolGallery unanimity, DATA F9; VERDICT 3-of-4 ≈0.995 precision, OCSR F18 [U]). Frontier-API voters join only after legal approves their output terms.
3. **Select.** Crops in the disagreement band (mean pairwise Tanimoto 0.6–0.9, DATA F5), plus crops with the MOSAIC tags the error feed flags.
4. **Annotate.** Annotators correct pre-annotations, about 30 s per crop instead of more than 3 min, and each crop is double-reviewed. Two chemists working 4 h/day produce about 5k crops/week (DATA F5).
5. **Freeze.** A human approves `pool_vN` with its manifest, license track, dedup report and a 200-item precision audit. recipe.py sees it only after a version bump in prepare.py.
6. **Gold hygiene.** Weekly cleanlab-style mining of model-vs-gold disagreements. Fixes are made only by humans, the gold is versioned, and the frontier is re-baselined (MET F21).

The agent cannot write to `data_engine/`. It can only file requests in `requests.md`.

## 9. Base model

**Inner loop: Qwen3-VL-2B-Instruct.**
- Apache-2.0, OCRBench 858, 256 tokens per 512² crop, and support in ms-swift, LLaMA-Factory and TRL (VLM F1, F2, F29, F32).
- Instruct rather than Thinking: Instruct is better at OCR (VLM F2).
- LoRA on attention plus MLP gate/up with the ViT frozen. Attention-only tuning learned +24.9 for −0.6 forgetting (VLM F25).
- No full SFT inside the loop: direct full SFT failed for MolSeek-OCR (OCSR F16).
- Small Qwen VLMs are proven in this domain. MarkushGlyph (Qwen3.5-2B LoRA) beat MarkushGrapher-2 (OCSR F26), and GTR-Mol-VLM (3B) matched MolScribe on real crops (OCSR F9).

**Promotion target: Qwen3-VL-8B-Instruct with LoRA** (OCRBench 896). If 2B→8B ρ stays below 0.5, the 4B model becomes the inner-loop model.

**Fallbacks:** InternVL3.5-4B (Apache; vision-tower LoRA reportedly helps InternVL far more than Qwen, DATA F7 [U]); Qwen3.5-4B with packing off (it thinks by default, needs flash-linear-attention, and DeltaNet state leaks across packed rows, VLM F4).

**Avoid:** Gemma 3 (fixed 896 px, VLM F7), AGPL models (MinerU2.5, ChemDFM-X) and NC weights (VLM F35).

## 10. Organization

**Nights 1–3:** a single greedy agent. Early gains concentrate: in Agora, the first 3 improvements gave 93% of the descent (AR F12).

**From night 4,** a shared ledger and four roles. Lineage is the key ingredient: 16 vs 3 valid improvements with and without it (AR F11).
- **Three recipe workers,** two GPUs each: depiction and degradation (items 2, 3, 8–10); chemistry and labels (1, 5–7, 13, 14, 18); mixture and selection (4, 15–17, 19, 20).
- **Critic,** from a different model family. It reads diffs and data cards, never eval data. It audits every PROMISING diff for leakage, probing and literals. When confirmed gains stay below 0.3 pp over 20 trials, it issues a 2–5 sentence "change region" directive (AR F15).
- **Frontier controller** (fixed code, not an LLM): keeps 6 elite recipes, picks parents by score plus novelty, allows recipe crossover (merging mixtures is natural for data), and forces novel exploration when one idea cluster exceeds 30% of trials (AR F12, F13).
- **Data-engine steward,** daytime only. It drafts pool releases for human approval.

**Memory.**
- **`ledger.jsonl`** rows record hypothesis, diff summary, parent, data card (pool shares, style entropy, stereo/abbreviation/Markush fractions, mean tokens), rung, all metrics and slice deltas, rejection bucket, receipt hashes, status and failure reason.
- **Compact slices.** Each agent sees the incumbent, its own last 15 rows and its siblings' last 5 (AR F11).
- **`failed_ideas.tsv`** is keyed by normalized recipe descriptors. A pre-flight novelty check rejects near-duplicates unless the hypothesis says what changed (AR F3, F15).
- **`results.tsv`** adds per-slice scores, NLL, guardrails, tokens, unique samples, datagen/train/eval seconds and the rejection bucket to autoresearch's columns.

**Human checkpoints.** Daily (20 min): review per-slice frontiers and NLL-vs-S correlation, then rewrite program.md; agents revise strategy in only 2.1% of transitions, and one redirect was worth up to 17 points (AR F30). Weekly: re-cut W_n, release pools. Humans also sign off stage exits and any change to metric, tiers or generators.

## 11. Anti-reward-hacking and integrity

Observed attacks so far: evaluator rewrites, training on test data, checkpoint swaps, seed lotteries and scorer loopholes (AR F4, F10, F29; MET F30–F33). Countermeasures:
1. **Isolation (§3).** The agent cannot read gold or run evaluation and emits only a recipe and KNOBS; evaluation loads the sha256-checked base plus a shape- and module-checked safetensors adapter.
2. **Receipts.** Each row stores sha256 of recipe.py, train.py, generators, shard manifest, adapter, harness and gold, plus RDKit version, token/step counts and adapter-delta norm. Zero-step or unchanged-weight runs are rejected (AR F4); unsigned rows are ignored.
3. **Leakage defense in depth (§8),** plus a nightly zero-overlap audit of the incumbent's full shard against every tier (InChIKey, block 1, pHash, family).
4. **Canaries.** 100 never-published molecules, rendered in eval-like styles, sit inside dev-fast. A dev gain that canaries do not share means memorization.
5. **Perturbation probe.** Every night, 10% of dev-fast is re-rendered or JPEG-compressed. A perturbation gap that grows by more than 3 pp raises an alert.
6. **Strict scorer.** Invalid structures count as wrong, duplicates as false positives, and missing documents as zero recall. Only the first prediction counts, and precision is part of the score. score.py already does this (MET F15, F16).
7. **No seed lottery.** The runner fixes the seed, and rung 2 requires a 3/3 win. Claudini fell back to seed search once it stalled (AR ADDED).
8. **Rate limits.** A confirm query budget, rounded feedback, and no item-level output from dev-fast.
9. **Audit.** The critic audits every promoted diff. An LLM judge never computes scores (MET F35).
10. **Environment.** No network, no API keys, a pinned lockfile, and no new dependencies without a human (AR F20, F29).

## 12. Top risks, mitigations, and pivot rules

| Risk | Signal | Mitigation |
|---|---|---|
| Data levers saturate; capacity/representation is the bottleneck (MolSeek-OCR plateaued at DECIMER level, OCSR F16) | <0.5 pp confirmed/night for 3 nights while the 8B jumps >5 pp | Give one worker model levers (Angle A); inner loop to 4B |
| Rung 1 does not predict rung 3 / 8B (AR F42) | ρ < 0.6 | Double rung-1 tokens; re-tune LR weekly at rung 3 |
| Leakage via pools or generators (USPTO overlap) | Canary or sealed-test gap | Family dedup; literature-derived primary slices |
| Tiny Markush sets (M2S has 103 items; DOC ADDED) | Wide CIs | Weight capped at 0.2; annotate 500 in-house Markush items by week 4 |
| License contamination | Manifest audit | Track tags; datagen refuses non-C; legal review |
| Gold errors (≥3%, MET F21) | Disagreement mining | Versioned gold, re-baseline |
| Blocklist probing | Rejection alerts | Silent replacement; critic |
| Synthetic SAR tables don't transfer (unstudied, DATA F38) | Real Stage-B slice flat | Shift weight to aligned real regions and annotation |

**Pivot rules.**
- **End of week 3.** Pivot if either holds:
  - the confirmed data-only gain on the real-literature slice is below 3 pp;
  - the 2B incumbent still trails MolScribe's 62.3 on MRBW-confirm.

  Then stop using the VLM as the structure recognizer. Hand structures to a permissively licensed specialist (MolScribe, MIT, or OCSRGlyph, Apache) and move the data loop to Stage B (records and linking), where VLMs have the edge (DOC F31).
- **Weak 8B correlation.** If ρ(rung 1, 8B) stays below 0.4 after two fidelity redesigns, replace autonomous structure search with weekly 8B data ablations designed by humans.

## 13. Six-week plan

| Week | Work | Deliverable |
|---|---|---|
| 1 | Containers, mounts and UIDs; runner, gates, receipts; score.py v1.1 (gaps 1–8) + 200 golden pairs; `uv run prepare.py --stage A` (Track-C pools, tiers, dedup report); generators v0; 60-s throughput probe to set TOKEN_BUDGET; W0 (about 4 h); 5-seed noise run; baseline rows (zero-shot, MolScribe, OCSRGlyph, MarkushGlyph, W0) | `runner.py submit` works end to end; noise report |
| 2 | Stage A nights 1–3 with a single agent; first 15 rung-1/rung-3 pairs → ρ; data-engine harvest and voters; annotation starts (target 3k) | ≥150 ledger rows, ρ report, per-slice frontier plot |
| 3 | Multi-agent setup with critic and frontier controller; `pool_v1` (about 50k consensus + 3k human labels); W1 re-cut; 8B transfer; first sealed run | Stage A report: go or pivot |
| 4 | score.py v1.2 (closed endpoint/target vocabularies, unit table, %inh concentration, ranges); SAR-table generator; BindingDB/ChEMBL alignment; record tiers (≥100 / ≥300 docs); W0-B baseline | Stage B harness and baseline |
| 5 | Stage B nights; `pool_v2` with SAR regions; weekly re-cut | Record-F1 frontier, precision guardrail holding |
| 6 | Full-scale 8B run of the best A+B recipe (about 2 GPU-days); sealed test for both stages; write-up of lineage, transfer and cost | Model card, data card, recommendation on Stage C or pivot |
