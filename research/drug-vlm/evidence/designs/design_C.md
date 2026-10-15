# Design C: A Scaled Research Org with a Multi-Fidelity Ladder and Verifiable-Reward RL

**Assumptions.** One node with 8x H100-80GB, 128 CPU cores (RDKit scoring and rewards) and 8 TB NVMe; training containers have no internet. The team will use the model commercially, so only the commercial-safe data track ships. Staff: an ML lead (owns program.md and the harness), a cheminformatician (owns gold and the scorer) and 2 part-time curators. Agents are Claude Code / Codex instances plus a critic from another model family. Today is 2026-10-15.

**Citation key.** AE = autoresearch-ecosystem, OS = ocsr-sota, DI = doc-chem-ie, VF = vlm-finetune-practice, DS = data-sources, ME = metrics-eval-design. † = [UNVERIFIED], used only as a prior.

## 1. Thesis

Karpathy's loop works because val_bpb is cheap, low-noise and protected by convention, and because one GPU, one incumbent and a greedy keep/discard rule are enough when the noise is around 1e-3 and the wins are around 1e-2. None of that holds here:
- A VLM extraction metric on 1-3k items has a standard error of about 1 pp.
- A night of about 100 trials produces a roughly 2 pp "win" from noise alone (ME F22, F24).
- In the closest public analogue, a VLM-LoRA dev gain halved and lost significance on a locked set (AE F9).
- Single-incumbent search collapses early (AE F12, F13).

My bet: with 8 GPUs the winning move is not 8x more greedy trials but a research org. Specialist agents feed a population frontier through a three-rung fidelity ladder (2B short, 8B medium, 8B full) whose cross-rung rank correlation is measured weekly. A deterministic, placebo-calibrated gate decides every promotion on data the proposer never saw.

Second, the existing scorer is a verifiable reward. InChIKey identity and order-invariant record F1 are what ChemVLR, COMO and Re3-DAPO show RL can climb when the reward matches the evaluation's equivalence notion (OS F20, F36; DI F13; ME F10). So the org has an RL lane that builds rewards from harness primitives on training labels only and must beat an equal-compute SFT control. Each of these three ingredients alone has a documented failure mode; together they compound.

## 2. Staged scope

The inner loop starts where the signal is cheapest and most verifiable, then widens. Detection stays frozen throughout; it is close to saturated on digital PDFs (DI F33).

| Stage | Unit scored | Inner-loop objective | R1 proxy | Exit criteria (all required, then human sign-off) |
|---|---|---|---|---|
| S1 (wk 1-4) | Single-structure crops, incl. Markush | S_crop: slice-weighted struct-F1 (InChIKey or CXSMILES key) | Qwen3-VL-2B, 20k crops | (a) ESTABLISHED 8B frontier ≥ 70.0 on the confirm portion of SMILES-evaluable MolRecBench-Wild (MolScribe 62.3, GPT-5.6-Sol 74.9; OS F10) and ≥ 55 CXSMILES-key EM on IP5-M (MarkushGrapher-2 53.2-53.7, MarkushGlyph 58.2; OS F26), or a plateau of 5 days without an ESTABLISHED promotion. (b) ρ-audit healthy (§6). (c) RL verdict recorded: beats SFT control, or retired. (d) Sealed-test within 3 pp of confirm. |
| S2 (wk 5-8) | Regions: SAR/R-group table crop + scaffold crop + nearby text → records | S_reg: existing composite (0.5·struct-F1 + 0.5·record-F1), slice-weighted | 2B, 6k region crops | Confirm record-F1 ≥ hybrid baseline + 15 pts (PaddleOCR-VL table crops + best commercial-safe specialist OCSR + zero-shot 8B linking); ID-link F1 ≥ 0.85. |
| S3 (wk 9+) | Whole documents (paper + SI, patents): cross-page tables, "Example N" coreference, provenance | S_doc: same composite at document level + provenance diagnostic | 2B, 2-page excerpts | Set by the human after S2; ~4 pages/s per H100 (VF F11) caps dev at ~300 pages. |

**Why crops first.** The end-to-end gap is in joining structures to values (BioMiner: value recall 0.85, structure 0.58, joint 0.33; DI F9), but joining can never beat structure accuracy. Crops also give the best signal-to-noise per GPU-second and the only large, permissively licensed real supervision: USPTO complex-work-unit (CWU) image/MOL pairs (DS F2, F3). Only exit criteria expand scope, never agents, and each stage keeps the previous scalar as a veto slice, so S2 cannot buy record-F1 by degrading OCSR.

## 3. File layout and the fixed/editable contract

```
research/drug-vlm/
  harness/            # READ-ONLY to agents (owner uid=harness, mounted ro) - the prepare.py analogue
    prepare.py        # builds pools/, eval splits, manifests, sha256; blocklists eval InChIKeys/families
    score.py          # existing scorer + v1 patches (§4.5)
    convert.py        # output converters: smiles|cxsmiles|gtr_cot|graph_json|records_yaml|records_json -> Mol/records
    reward.py         # RL primitives reusing score.py normalization (identity, skeleton, validity, record_f1)
    evaluate.py       # eval container only: base+adapter, harness-owned vLLM decoding, signed receipt
    gate.py           # deterministic statistics: R1/R2/confirm gates, placebo bookkeeping, shrinkage
    ladder.yaml       # rung definitions: model, sample budget, caps, eval splits
    golden_pairs.jsonl# ~200 tricky structure pairs; verdicts must not change on any RDKit bump
  recipe/             # THE ONLY AGENT-EDITABLE PATHS - the train.py analogue
    train.py          # data mixture, curriculum, adapter config, optimizer, SFT loop, optional RL/MRT phase
    infer.yaml        # prompt template, output_format (from convert.py registry), max_pixels, grammar id
    NOTES.md          # hypothesis + expected effect (required; parsed into lineage)
  org/                # program.md analogues, HUMAN-EDITED
    program.md        # shared contract, scoring rules, simplicity criterion, data/legal policy
    roles/{director,data,repr,optim,rl,critic,fixer}.md
  orchestrator/       # HUMAN-OWNED: scheduler.py, frontier.py, rho_audit.py, placebo.py, `orch` CLI
  lineage/            # APPEND-ONLY, orchestrator writes: lineage.db (SQLite), results.tsv export, receipts/
  pools/  evalsets/   # pools: ro in train containers; evalsets: mounted ONLY in the eval container
```

```
# human, once per pool version
uv run harness/prepare.py build --pools v1 --track commercial && uv run harness/prepare.py freeze --version v1
# agent
orch submit --parent est-07 --lane data --rung R1     # pushes recipe/ diff, queues R0 -> R1
orch slice --me                                        # compact lineage slice for this agent
# orchestrator only (agents cannot invoke)
python harness/evaluate.py --trial t1234 --split devA --stage 1
python harness/gate.py r1 --trial t1234 ; python orchestrator/rho_audit.py --week 2026-W44
```

These rules are enforced by mechanism, not by instruction (AE F4, F10; ME F29).

- **Path allowlist.** Agents push to a bare repo whose pre-receive hook rejects:
  - any commit touching paths outside `recipe/`;
  - any `train.py` longer than 1,500 lines;
  - imports of `socket|requests|urllib|http|subprocess|ctypes`;
  - literal strings that hit a Bloom filter of eval InChIKeys, SMILES and document IDs.
- **Process isolation.** Training runs as uid `agent` with `pools/` read-only, `/scratch` (200 GB quota), no network, no API keys and a scheduler GPU lease. `evalsets/` and gold exist only in the eval container (uid `harness`). It verifies the base-model sha256, decodes with harness-owned settings (greedy, fixed `max_new_tokens`, n-gram repetition guard), converts with `convert.py`, scores with `score.py`, and HMAC-signs a receipt: sha256 of train.py, infer.yaml, adapter, harness and manifests, plus optimizer steps, samples_seen, ‖ΔW‖ and wall time. Lineage rejects rows without a valid receipt; the score never comes from agent stdout.
- **Data access** goes only through `harness.datapool.reader(pool, ids)`, which logs every sample ID. Runs whose samples_seen or steps deviate from the receipt by more than 2%, or with ‖ΔW‖ = 0, are rejected.
- **Converters and reward primitives** are registry-only. A new output representation is a Harness-Steward ticket (cheminformatician, with golden-pair tests), never an agent commit.

## 4. The metric

### 4.1 Primary scalar

In Stage 1, S_crop = Σ_s w_s · structF1_s over four fixed slices:
- **Real literature, 0.40:** MolRecBench-Wild, BioVista-OCSR, ACS, in-house papers and SI.
- **Real patents, 0.20:** in-house patents, corrected CLEF/JPO, a USPTO-10K-abb subset.
- **Markush, 0.20:** scored on the CXSMILES key.
- **Hard, 0.20:** gold with stereo, MolRecBench-Wild subset C, low-resolution crops.

Fixed weights stop the agent trading Markush or stereo accuracy for easy-image gains (OS implications). structF1 is score.py's multiset key F1; on single-structure crops it is exact match with a precision penalty. Stages 2-3 keep score.py's per-slice 0.5·struct-F1 + 0.5·record-F1, then slice-weight.

### 4.2 Representation independence (the val_bpb property)

val_bpb is per byte, so vocabulary changes compare fairly; here everything is scored per molecule identity. Whatever registered format infer.yaml declares (SMILES, GTR-style traversal CoT, atom/bond JSON, CXSMILES, YAML records), `convert.py` maps the answer to an RDKit Mol, applies FragmentParent+Uncharger and emits a standard InChIKey (canonical CXSMILES for dummy atoms) for score.py. Unconvertible output is an invalid prediction, never a raw-string compare (ME F2). Budgets count samples seen, so a verbose representation pays in wall clock against the cap.

**Companion signal: MI8**, the mean identity rate over 8 samples at T=0.7 on 400 dev-A crops. It is computed after conversion (representation-independent), has lower per-item variance than greedy 0/1 (ME F26) and equals the RL objective's expectation. Teacher-forced bpb is logged only within one representation family.

### 4.3 Guardrails (veto regardless of S)

Veto if: invalid-structure rate +0.5 pp; record precision (hallucination proxy) −1 pp; orphan-record rate +1 pp; any slice drops more than max(1.5 pp, 2·SE_slice); stereo gap (struct_f1_nostereo − struct_f1) widens > 1 pp; general-capability probe (200 DocVQA/OCRBench/ChartQA items) −2 points (VF implications); OOD perturbation gap (10% of dev re-rendered with JPEG, rescaling, scan noise) widens > 2 pp; canary accuracy diverges from dev > 3 pp; eval wall clock > 2x baseline; parse failures > 1%; VRAM > 75 GB.

### 4.4 Eval-set composition

All splits are by article DOI or patent family, never by image.

| Set | Sources (license) | dev-open | dev-A (R1) | dev-B (R2) | confirm-C | sealed-T |
|---|---|---|---|---|---|---|
| Real-literature crops | MolRecBench-Wild revised, 2,392 SMILES-evaluable (no license on the card; CC-BY source papers; eval-only). BioVista-OCSR, 8,861 (MIT). ACS, 331 (Keylab, mixed). | 200 | 700 | 1,400 | 1,400 | - |
| Patent crops | Keylab corrected CLEF 992 and JPO 449; USPTO-10K-abb 1,000 (mixed; eval-only) | 100 | 300 | 600 | 600 | - |
| Markush | IP5-M 878, M2S 103, USPTO-Markush 74 (MarkushGrapher sets: CC BY 4.0; IP5-M card untagged); in-house 300 | 50 | 250 | 450 | 450 | 150 |
| In-house crops | Team's own SI and patents, double-annotated and adjudicated | 100 | 250 | 550 | 500 | 1,500 (time-split: docs published on or after 2026-07-01) |
| Records (S2+) | In-house 200 docs (~6k records). ChemX Oxazolidinones and Benzimidazoles (MIT). BioVista triplets, val/test split (MIT labels; PDFs fetched per PDB ID). | 30 docs | 40 | 60 | 50 | 50 + 30 EPO/CNIPA |

With ~8% discordance and design effect 1.2 (~3 crops per article), paired SE_diff is ≈ 0.8 pp on dev-A (1,500) and ≈ 0.57 pp on dev-B (3,000) (ME F22). Agents see per-item errors on dev-open only; dev-A/B as aggregates rounded to 0.1 pp; confirm-C as pass/fail; sealed-T never.

### 4.5 Gaps in the existing score.py (patched by humans in harness v1)

1. **No strata**: a micro-average lets SAR-heavy patents dominate. Add `--strata strata.tsv` with fixed weights and per-slice output.
2. **Bootstrap clusters on doc_id**: pages of one family or paper are correlated. Resample by `cluster_id` (DOCDB family/DOI), hierarchically with seeds.
3. **No gate**: `paired_compare` gives a 90% two-sided percentile CI without multiplicity control, seeds or shrinkage. That becomes `gate.py`.
4. **Value tolerance**: `VALUE_REL_TOL=1e-3` rejects faithful rounding. Verified: gold pIC50 7.2 → 63.096 nM vs printed "63 nM" → False. Compare in log10 within 0.01 units or at reported significant figures.
5. **Thin measurement semantics**: target is casefolded text ("EGFR" ≠ "EGFR L858R/T790M" ≠ "ErbB1"), so add a frozen synonym→UniProt table and a variant field. No assay type, species or cell line; "%inh @ 10 µM" loses the concentration; ranges, "NT/inactive" and ± SD unsupported; µg/mL needs MW; ADMET units become literal strings (verified: "mL/min/kg" ≠ "ml·min-1·kg-1"); "≤" collapses into "<".
6. **Markush**: canonical-CXSMILES equality without R-table, m/Sg semantics or R-label mapping (`*c1ccccc1` and `[1*]c1ccccc1` key differently). Add substituent-table F1 and deterministic parsed-graph equality, no wall-clock MCS (OS F24; ME F11, F13).
7. **Unscorable gold silently dropped**: 2,632 of 5,024 MolRecBench-Wild images have no SMILES; report a graph-track diagnostic.
8. **No identifier-structure check**: add `id_struct_f1` (label + key); linking is the S2 bottleneck.
9. **Toolkit drift**: RDKit version is printed, not enforced. Pin RDKit/InChI and fail on any `golden_pairs.jsonl` verdict change (ME F7); add an "adaptive" stereo rule for flat-drawn racemates and enhanced stereo.
10. **No per-item API or size caps**: expose per-sample components via `reward.py`; cap 300 atoms per structure and 2,000 compounds per document so adversarial outputs cannot stall scoring.

## 5. Budget per experiment

**Throughput anchors:** training on one H100 at ~10-45 crops/s for Qwen3-VL-2B and ~3-13 for 8B-LoRA (VF F33†, derived); MolSight 8B LoRA ≈ 1.5 samples/s per A100 (VF F19); vLLM 7B ≈ 3,050 output tok/s per H100, ~30 crops/s (VF F11, F34†); a 512² crop is 256 vision tokens in Qwen3-VL (VF F32). A 60-second probe in week 1 replaces these; budgets are then frozen in `ladder.yaml`.

| Rung | Model / GPUs | Train budget | Arithmetic | Eval | Wall | Rate |
|---|---|---|---|---|---|---|
| R0 smoke | 2B, 1 GPU | 60 s | Checks NaN, OOM and throughput; 100-crop eval | 1 min | 3 min | gate only |
| R1 screen | Qwen3-VL-2B, 1 GPU | 20,000 crops (cap 20 min) | 25 crops/s gives 800 s ≈ 13 min | dev-A 1,500 greedy (~25 s), MI8 3,200 generations (~40 s), load/merge ~2 min | ≈ 18 min | 3.3/h/GPU |
| RL-R1 | 2B, 1 GPU, colocated vLLM | 150 GRPO steps × 32 prompts × 8 rollouts | 256 rollouts × ~120 tok ≈ 31k tok. Generation ≈ 6 s (assumes ~5k tok/s for 2B, scaled from VF F11), update ~10 s, ~20 s/step with overhead, ≈ 50 min | as R1 | ≈ 55 min | 1/h |
| R2 promote | Qwen3-VL-8B LoRA, 1 GPU | 60,000 crops (cap 150 min) | 8 crops/s gives 125 min | dev-B 3,000 at ~30/s ≈ 2 min, load 3 min | ≈ 130 min | 0.46/h/GPU |
| RL-R2 | 8B LoRA, 4 GPUs (2 rollout, 2 train) | 200 steps × 64 prompts × 8 rollouts | 61k tok/step: ≈ 10 s generation + ~32 s training, ≈ 2.4 h total | dev-B | ≈ 2.6 h | at most 1/day |
| R3 full | 8B, 4 GPUs DDP | ≈ 500k samples (+ optional RL phase) | 4 × 8 crops/s = 32/s gives 4.3 h | dev-B + confirm-C | ≈ 5 h | nightly |

**GPU allocation:**

| GPUs | Day | Night (22:00-03:30) |
|---|---|---|
| 0-3 | R1, 24/7: 12 trials/h, about 120 per 10-hour night, about 280 per day | R1 continues |
| 4-5 | R2: 2 parallel jobs, about 12 per day including ρ-audit jobs | R3 on the frontier's top member, then the confirm-C eval |
| 6 | RL-R1, about 14 per day | R3 |
| 7 | Eval service: confirm-C, canaries, perturbations, placebos | R3 |

With re-seeding, a candidate averages ~1.4 R1 runs: ~85 SFT candidates per night, 4-6 R2 promotions per day, one R3 per night. Single-GPU autoresearch ran ~12 experiments/h (README); one R1 GPU here manages ~3/h, which is why the parallel org exists (AE F6, F11).

## 6. Keep/discard rule under noise

**Calibration (week 1).** 8 baseline seeds at R1 and 3 at R2 give σ_seed per rung; doc-cluster bootstrap gives SE per split; 20 placebo R1 runs (baseline, new seed, cosmetic diff) give the null Δ distribution directly. Seed effects are large (ME F27), and autoresearch's own final "win" was a seed change (AE F3).

**Gates** (all in gate.py, all deterministic):
- **R1 screen**: one candidate seed vs the incumbent's 8 banked seeds. Pass if Δ_A ≥ max(1.0 pp, 1.25·SE_tot) with SE_tot = √(SE_diff² + σ_seed²(1+1/8)), MI8 Δ has the same sign, and no veto. Passers get 2 more seeds and become **PROMISING** if the hierarchical (cluster × seed) bootstrap one-sided 95% lower bound of the 3-seed mean Δ_A is above 0 (speedrun-style funnel; AE F8).
- **R2** runs on dev-B, which never informed R1 selection: re-seeding cannot remove the item-level winner's curse because all seeds share dev-A's items; only fresh items can. One candidate seed vs the incumbent's 3 banked R2 seeds; pass if Δ_B ≥ 0.5 pp, the one-sided 95% lower bound is above 0, and no veto. A pass earns a second seed.
- **ESTABLISHED (confirm-C)**: 2-seed mean vs the current established member; one-sided 97.5% lower bound above 0 and no slice regressing more than 2 pp, the locked rule of the Qwen3-VL analogue (AE F9). C reveals pass/fail only, Ladder-style (ME F25†), has a 40-query budget per version, and is refreshed with newly annotated documents when the budget is spent or monthly.
- **Release**: weekly the human scores the top-3 frontier members plus incumbent on sealed-T (top-k beats dev-best; AE F16) and soups the winner's seeds (VF F12).

**Shrinkage.** R1 Δs are shrunk by empirical Bayes against the running distribution of all R1 Δs (mostly ~0) before ranking, and agents see shrunk values. This stops them chasing the max-of-85 artefact, ≈ 2.45 SE or ~2 pp on dev-A (ME F24).

**Placebo arms.** The orchestrator hides a placebo in 1 of 10 R1 slots and 1 R2 slot per week. Target pass rates: R1 screen ≤ 20%, PROMISING ≤ 3%, R2 ≤ 2%, confirm ≤ 1%. If a rolling 50 exceeds them, gate.py tightens its multiplier and pages the human, so "is our gate honest?" is a measured number.

**Cross-fidelity ρ-audit** (rho_audit.py, weekly). 16 R1-scored candidates spread over the Δ_R1 range (4 top, 8 middle, 4 bottom incl. discards, avoiding range restriction) go to R2. Report Spearman ρ, Kendall τ and pairwise sign concordance with bootstrap CIs, overall and per lane. Healthy: ρ ≥ 0.6 and concordance ≥ 75% (DataDecide: small-scale rankings pick the large-scale winner ~80% of the time; AE F41). A failing lane becomes R1-ineligible and goes straight to R2. LR, schedule and resolution changes are R1-ineligible from day 1 because of short-horizon bias (AE F42); R2 runs a fixed 3-point LR sweep (×0.5, ×1, ×2) for any change to trainable modules. Monthly, the same audit runs R2→R3.

## 7. The agent's search space (prioritized)

| # | Lane | Idea | Rung | Expected impact | Evidence |
|---|---|---|---|---|---|
| 1 | Data | Real-data fraction 0/10/30/50% (USPTO CWU + in-house + born-digital) | R1 | Largest single lever: +10-30 pp on real literature | DS F7†, OS F17†, DS F3 |
| 2 | Data | Labels that keep abbreviations as drawn (superatoms) vs fully expanded | R1 | Large on the abbreviation slice (MolScribe: <20% → ~70% after relabeling) | OS F9, F29; DI F18 |
| 3 | Data | Augmentation strength plus a weak-to-strong curriculum | R1 | +29 pp OOD in MolParser (40.1 → 69.5 → 76.9) | DS F6 |
| 4 | Data | 2x chiral oversampling (Stereo-200k) plus wedge↔wavy swaps | R1 | Stereo slice +3-8 pp | OS F26, F30 |
| 5 | Data | COMO-style remap of '*' to the visible label in USPTO MOLs; teacher-agreement filter | R1 | Less label noise | DS F2; OS F8 |
| 6 | Data | Hard-example mining from the teacher-disagreement band | R1 | Data efficiency | DS F5 |
| 7 | Data | 10-30% general-document replay | R2 | Protects the forgetting probe | VF F25 |
| 8 | Repr | GTR-style graph-traversal CoT then SMILES, vs direct SMILES | R1→R2 | +2-3 pp | OS F9 |
| 9 | Repr | Atom/bond JSON with coordinates, deriving stereo from geometry | R2 | Stereo slice | OS F28, F30 |
| 10 | Repr | CXSMILES plus R-table output for Markush crops | R1 | Markush slice toward 55-60 | OS F26 |
| 11 | Repr | max_pixels from 256 to 1,024 tokens per crop; aspect-ratio padding | R2 | Small and low-resolution slice | VF F32; OS F31 |
| 12 | Repr | YAML vs JSON records; grammar-constrained decoding | S2 R1 | Fewer parse failures and loops | VF F13, F14 |
| 13 | Optim | LoRA targets (attention only, MLP gate/up, or all linear layers); rank 64-128 | R1 | ±2 pp; affects forgetting | VF F25; OS F26 |
| 14 | Optim | Unfreeze the top-k ViT blocks at 0.1x LR | R2 | Depends on the base model | VF F25; DS F7† |
| 15 | Optim | LoRA, then selective full fine-tuning with split LRs | R2 | Stability | VF F17 |
| 16 | Optim | LR and batch sweep per trainable set | R2 only | 1-3 pp | VF F28; AE F42 |
| 17 | RL | DAPO/GRPO with a binary full-InChIKey reward + 0.1 validity; keep only prompts whose 8 rollouts disagree | RL-R1/R2 | +3-8 pp over SFT | OS F20, F36 |
| 18 | RL | COMO-style interleaved MRT (N=8-16, T=0.5, R = 0.1v + 0.5·sim·v + 0.4·exact, λ=0.1) | RL-R1 | 0 to +16 pp vs the MLE backbone (large on JPO/ACS, near 0 on WildMol) | OS F8; ME addendum |
| 19 | RL | Stereo-aware shaping: −0.3 when the skeleton matches but the stereo layer differs; penalize stereo invented on achiral gold | RL-R1 | Chiral slice | OS F30; DI F34 |
| 20 | RL | Record-level RL: reward = per-document record-F1 from score.py (S2) | S2 RL | +3-4 F1 | DI F13; VF F22 |
| 21 | RL | DPO on about 500 (correct, plausible-wrong) pairs mined from teacher disagreements | RL-R1 | Markush and chirality errors | OS F7 |
| 22 | RL | KL 0.01 vs an MLE anchor; repetition guard; seed souping | RL-R2 | Avoids collapse | VF F12, F22, F26 |
| 23 | S2 | Identifier visual prompts (IdtVP) vs indexed boxes for linking | S2 R1 | +5-7 F1 zero-shot | DI F12, F13 |
| 24 | S2 | VLM reads the scaffold and R-table; RDKit enumerates deterministically | S2 R1 | R-group records | DI F23 |
| 25 | Meta | AutoData-style selection programs over pool feature columns | R1 | Recipes that transfer | AE F39 |

Known negatives go into program.md as priors: direct full SFT of an OCR VLM diverged, GSPO with a graph-heavy reward lowered exact match (OS F16; ME F10), and RL without SFT does nothing (OS F20). Every RL candidate carries an equal-GPU-hour SFT-continuation control (COMO's ablation; OS implications).

## 8. Data strategy

**Fixed by prepare.py** (versioned `pools/vN`; only humans bump versions).

*Commercial-safe track (ships):*
- **USPTO CWU pairs**: grant and application TIF+MOL/CDX, 2001-2024, ~600k after cleaning; patents with 37 CFR 1.71 notices dropped (DS F1†, F37†).
- **Renders**: 2M PubChem/ChEMBL SMILES via RDKit (BSD), Indigo (Apache) and CDK (LGPL, separate tool) in RanDepict-style random styles, abbreviations kept as drawn (DS F31).
- **Permissive sets**: MarkushGrapher-Datasets ~235k (CC BY 4.0); Stereo-200k (Apache); MolScribe USPTO/PubChem data (HF tagged MIT; DS F3); glyph-datasets (Apache) **minus its MolParser-derived rows**.
- **Born-digital labels**: ChemScraper vector extraction on the PMC OA commercial subset and CC-BY ChemRxiv (DS addendum, F33†, F34†).
- **In-house crops** (training split only).
- **S2**: synthetic SAR tables on perturbed real layouts from BindingDB patent rows (CC BY) and ChEMBL rows (CC BY-SA: share-alike if redistributed), never generic synthetic pages (DS F10, F38); BindingDB↔USPTO-XML and ChEMBL compound_key↔OA-table alignments (DS F19†, F20†).

*Research-only track (separate leaderboard, never in shipped weights):* MolParser-7M, MolGallery, MolParser-Mobile and MolDetv2 as teachers, COMO weights (NC), MolMole_Patent300 (NC-ND, eval-only) (DS F4; OS F38). Every manifest row carries `license`, `track`, `source_id`, `family_id`; prepare.py refuses to mix tracks.

**Label fidelity.** A USPTO pair is kept only if ≥ 3 of 4 commercial-safe recognizers (MarkushGrapher-2, OCSRGlyph, MolScribe, current frontier) agree with the MOL on identity; the rest go to curation (agreement beats re-render checks, DS F8†; MolGallery's unanimity filter, DS F9). Eval gold is double-annotated with pre-annotation (~30 s/molecule; DS F5) and adjudicated; weekly disagreement mining feeds versioned gold fixes.

**Dedup and leakage.** Splits by DOCDB family and DOI. Every eval InChIKey block-1, and the Bemis-Murcko scaffolds of SAR documents, is blocklisted from all training pools and generator SMILES. pHash removes near-duplicate images. Classic-benchmark source patents are excluded (DS F30; ME F28†). sealed-T is time-split (published ≥ 2026-07-01, after base-model and pool snapshots). A nightly audit intersects logged sample IDs with the blocklists.

**Data engine.** Teacher ensembles, active learning in the 0.6-0.9 agreement band and annotation rounds run offline under a human-owned program and produce new pool versions.

**Agent-tunable (train.py):** per-pool sampling weights, curriculum, augmentation parameters within registered ranges, oversampling rules, and selection programs over precomputed features (difficulty, teacher agreement, atom count, stereo flags).

## 9. Base model choice

**Target: Qwen3-VL-8B-Instruct**: Apache-2.0, OCRBench 896, 32 px per visual token, supported by ms-swift, LLaMA-Factory, TRL, verl and EasyR1 (VF F1, F2, F29); Instruct beats Thinking on OCR (VF F2). **R1 proxy: Qwen3-VL-2B-Instruct**, sharing tokenizer, vision tower and chat template, so recipes should transfer like Karpathy's d12→d24 changes (AE F1); Qwen3-VL-4B is the fallback proxy if the ρ-audit fails. **Not Qwen3.5/3.6 as primary**: thinking-by-default and the DeltaNet packing hazard make automated search fragile (VF F4, addendum); Qwen3.5-2B stays a Markush side branch given MarkushGlyph (OS F26). **Fallbacks**: InternVL3.5-8B (Apache; vision LoRA reportedly helped it, DS F7†) and Gemma 4 12B/E4B (Apache; 70-1,120 token budgets; VF F7 correction). Avoid AGPL/NC bases such as ChemDFM-X and MinerU2.5 (VF F35). **Memory**: 8B LoRA SFT, 2B full FT and 2B RL each fit one 80 GB card; 8B RL needs LoRA on ≥ 2 cards (VF F30). **Hybrid fallback**: if the VLM cannot reach specialist crop accuracy (§12), Stage 1 uses OCSRGlyph and MarkushGrapher-2 (Apache) as tools and the VLM loop moves to linking and tables, where hybrids win (DI F31).

## 10. Organization

| Role | Model | Owns | Cannot |
|---|---|---|---|
| Director (1; long-lived coordinator, Arbor-style) | strongest available | Lane quotas, weekly hypotheses, crossover requests, stage-exit proposals | Run jobs; edit the recipe |
| Lane specialists (4: Data, Representation, Optimization, RL) | mid-tier coding agent, each in its own git worktree | Proposing a diff plus a NOTES.md hypothesis on a frontier parent; reading their own lane's results | See dev-A/B items; see confirm results beyond pass/fail |
| Code Fixer | cheap model | One pre-execution review of every diff (OOM, shapes, schedule bugs) | Change the hypothesis |
| Critic/Auditor | different model family | Audits every R2-passer's diff and trace for leakage or eval contact. On stagnation, issues a 2-5 sentence "change region" directive, cleared at the next ESTABLISHED. | Vote on scores |
| Statistician | deterministic code (gate.py) | Every keep/discard decision | - |
| Harness Steward | cheminformatician (human) | Converter and reward registry; gold versions | - |

**Lane quotas** start at Data 35%, RL 25%, Representation 20%, Optimization 20% of R1 slots and are re-weighted weekly by valid improvements per GPU-hour, a bandit over lanes (AE F14, F23). Each specialist makes ~3 proposals/h; each passes a novelty check (normalized change descriptors and hypothesis embedding vs the tried-and-failed table), and near-duplicates are rejected unless NOTES.md states what differs (AE F3, F23).

**Lineage** (`lineage.db`): trial_id, parent_ids (crossover), lane, hypothesis, change_descriptor, diff_sha, rung, seeds, every metric component with CI, shrunk Δ, verdict (screen-fail / PROMISING / R2-pass / ESTABLISHED / veto:\<guardrail\> / crash), failure_reason, GPU-hours, receipt hash. An *insights* table holds claims that become usable only after a non-author agent reproduces them (Agora's verifications; AE F12). Each agent's context gets a compact slice: frontier rows, its own last 10, the adjacent lane's last 5. In CMU's ablation lineage separated 3 from 16 valid improvements per 200 trials (AE F11).

**A population, not an incumbent.** The frontier is a MAP-Elites archive of ≤ 8 ESTABLISHED members keyed by representation × paradigm (SFT, SFT+MRT, SFT+RL); R2-passers wait in a crossover pool. Parents are chosen by diversity-aware UCB with 50% exploit / 30% explore-known / 20% explore-novel slots (Agora AE F12; GEAR AE F13). Crossover: the Director merges two R2-passed diffs from different lanes, entering at R2 because interactions show at scale (AE F6).

**Stagnation triggers.** No ESTABLISHED in 36 h or 40 R1 trials → Critic directive and explore-novel slots raised to 50%. More than 30% of the last 50 trials in one hypothesis cluster → novel-only for 12 h. Frontier top-1 unchanged over 3 R3 nights → human paged. The first night still runs a single greedy agent, because early gains are front-loaded (AE F12).

**Human checkpoints.** Daily 08:30, 15-minute review of the nightly report (frontier per slice and rung, ρ, placebo rates, gate log, Critic flags). Monday rewrite of org/program.md and role files. Only the human changes metric, eval tiers, pool versions, model size or license track, and only the human runs sealed-T. Agents revise strategy in 2.1% of transitions while one mid-run redirect was worth up to 17 points (AE F30), so the daily review is budgeted, not optional.

## 11. Anti-reward-hacking and integrity

1. **Isolation**: gold never mounted where agent code runs; scoring in a separate container from adapter weights only; signed receipts bound to code, data and weight hashes (ME F29, PostTrainBench addendum; AE F4).
2. **Harness-owned decoding**: greedy, fixed token caps, repetition guard, no best-of-N on the main track (score.py already enforces first-prediction-wins); majority voting only as a separately labeled compute track.
3. **Weights and network**: only adapters on the sha-pinned base; no network, API keys or downloaded checkpoints (AE F29; ME F30).
4. **Mechanical checks**: steps, samples_seen and ‖ΔW‖ must agree between pool reader and receipt (AE F4).
5. **Leakage scans**: nightly manifest audit against eval InChIKeys, families and pHashes; Bloom-filter diff scan; generators refuse eval molecules.
6. **Canaries and perturbations**: never-published synthetic canary documents and nightly perturbation re-renders; divergence vetoes (ME implications).
7. **RL-specific**: rollout prompts only from training pools via the logged reader; rewards built from `harness.reward` primitives so normalization matches the scorer; the scorer is strict where published ones are lenient (qualifiers scored, duplicates are FPs, invalid rows count in precision, no ';' expansion; ME F15, F16); monitor exact-vs-graph divergence (ME F10) and repetition collapse on the OOD scan slice (VF F22).
8. **No LLM judge** in any scalar (ME F35); the Critic only flags, and flagged trials need human clearance before ESTABLISHED.
9. **Limited feedback**: scores rounded to 0.1 pp, per-item errors only on dev-open, confirm as pass/fail; placebo arms detect gate leakage.
10. **Resource caps**: 200 GB scratch and 2 GB adapter per trial, GC of discarded trials after 72 h, kill at 1.5x rung cap (AE F20).

## 12. Top risks, mitigations and pivot triggers

| Risk | Mitigation | Abandon or pivot when |
|---|---|---|
| 2B-to-8B rank correlation is poor (no public measurement exists for chemistry VLMs) | Per-lane ρ, the 4B proxy, R1-ineligible lanes, a longer R1 | ρ < 0.4 after two recalibrations: drop R1 and run R2 only, at about 12 candidates per day |
| VLM crop OCSR plateaus below the specialists (MolSeek-OCR; DI F31) | GTR-CoT or graph outputs, real data, MRT | End of week 4 with the 8B frontier more than 10 pp below OCSRGlyph on real-literature dev-B and improving less than 1 pp/week: switch to the hybrid (specialist tool, VLM for linking and tables) |
| RL adds nothing over SFT at equal compute (OS F16; ME F10) | Identity reward, difficulty filter, SFT control | 3 well-formed RL configs fail R2 against the control: retire GRPO, keep only MRT and DPO |
| Confirm set overfits through adaptive reuse | Pass/fail only, 40-query budget, refreshes, sealed-T | sealed-T trails confirm by more than 3 pp twice: freeze promotions and rebuild C |
| Gold-label bottleneck (200 record docs is about 400-600 curator-hours) | Pre-fill from BindingDB/ChEMBL; pre-annotation | Fewer than 120 docs by week 6: start S2 on ChemX and BioVista only, and postpone the S2 exit |
| License contamination | Per-sample manifests; pools separated by track | Any NC sample found in a commercial run: invalidate that run and all its descendants |
| Infrastructure fragility (in the Amazon deployment, 11 of 30 iterations were lost to timeouts; AE F15) | Code Fixer, R0 smoke, watchdog, idempotent eval | - |
| Agents gaming new attack surfaces | The §11 controls, placebo rates, the Critic | Any confirmed hack: freeze the lane, patch the harness, re-baseline |

## 13. Week-by-week plan (first 6 weeks)

- **W1 (Oct 19-25): Harness and baselines.** Containers, uid split, pre-receive hook, signed receipts; score.py v1 patches (strata, cluster bootstrap, log tolerance, golden pairs) plus convert.py, reward.py, gate.py; pools v1 (2M renders, 2019-2024 USPTO slice, CC BY Markush sets, Stereo-200k); eval splits from public sets plus the first 800 in-house crops. Baselines in `results.tsv` on dev-A/B: zero-shot Qwen3-VL-2B/8B, OCSRGlyph, MarkushGrapher-2, MolScribe, one frontier API model. 60-second throughput probes, 8-seed R1 and 3-seed R2 noise runs, 20 placebos. *Deliverable:* noise report (σ_seed, SE per split, null Δ distribution) and frozen `ladder.yaml`.
- **W2: Single-agent ladder.** One greedy agent on R1 for two nights, then R2 promotions; first R3 night; first ρ-audit (16 candidates). Curators start record gold, pre-filled from BindingDB/ChEMBL. *Deliverable:* first ESTABLISHED SFT recipe, ρ with CI, measured gate false-positive rates.
- **W3: Turn on the org.** Director, 4 specialists, Code Fixer, Critic; MAP-Elites frontier, novelty checks, stagnation triggers. RL lane: RL-R1 GRPO with identity reward vs SFT control, plus MRT. *Deliverable:* ~600 R1 trials, ≥ 2 ESTABLISHED members from different lanes, first RL-vs-control verdict.
- **W4: Stage-1 exit review.** RL-R2 on 8B, crossovers, first confirm-C refresh, sealed-T on the top 3; hybrid-pivot decision per §12. *Deliverable:* S1 exit memo (per-slice scores with CIs, ρ history, RL decision, license audit).
- **W5: Stage-2 setup.** Scorer extensions (target synonyms, %inh concentration, ranges, ADMET units, id_struct_f1, Markush table F1); region pools (synthetic SAR tables on real layouts, BindingDB/ChEMBL alignments); PaddleOCR-VL region-cropping baseline; record-F1 reward primitive. *Deliverable:* S2 harness v2 and hybrid baseline numbers.
- **W6: Stage-2 loop.** Org restarts on S_reg with S_crop as a veto slice; record-level RL lane; IdtVP and enumeration experiments; first S2 sealed-T. *Deliverable:* milestone report (frontier, sealed-T CI, cost per GPU-hour of improvement, go/no-go for Stage 3).
