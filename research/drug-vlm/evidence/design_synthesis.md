# 最终方案（英文原文）：药物文献抽取 VLM 的 autoresearch 循环

本文件是 `../README.md` 第 6 节的英文底稿：3 个设计 agent 各自给出方案（A 最小忠实移植、B 数据中心、C 规模化研究组织 + RL），3 个评委 agent 从可行性、评测严谨性、化学领域正确性打分后，由一个综合 agent 写成。生成日期：2026-10-15。

阅读时注意：
- 引用写法与 README 不同：(eco/ocsr/ie/vlm/data/met Fxx) 分别对应 README 中的〔ecosystem / ocsr / doc-chem-ie / vlm-ft / data / metrics Fxx〕，指向 `verified_findings.md`；[U] = 未核实。
- 文中的 "score.py v1.4" 是写作时的版本。附录 (c) 列出的缺口中，S1 第 1–3 条（来源加权与分层 bootstrap、按切片/护栏的配对增量与标准误、金标与评分器哈希）和 S2 第 7 条（"12 ± 3"、"10 µM"、"1.2 × 10⁻⁵"、列头单位）已在 v1.5 修复；v1.5 还修复了记录侧的对称排除、游离金属离子去盐、二分图最大匹配，并增加了 `--alpha` 和难度标签 `tags`。其余条目仍待完成。
- 少数吞吐与成本数字是推导值（如 LoRA 推理约慢 1.35 倍、2B 训练 25 张/秒），需要在准备阶段实测。

---


Citations: (eco/ocsr/ie/vlm/data/met Fxx) = digest topics; [U] = unverified, used only as a prior. "score.py" = harness v1.4 (48 tests green); proposed changes = v1.5.

**Assumptions.** One 8×H100-80GB node (≈128 cores, ≥8 TB NVMe); a 1×H100 fallback is in §5. Commercial use: only Track-C (commercial-safe) data and weights enter shipped models. Staff: ML lead, ML engineer, cheminformatician (1.0 FTE each), two chemists at ~65%, legal on call. One Claude Code agent in a sandbox.

## 1. Thesis

Start from Design A: a faithful port of autoresearch's four mechanics (one editable `train.py`, a fixed budget, one representation-independent scalar, a keep/discard ratchet), aimed first at crop-level structure recognition. Structures are the join key of every record: BioMiner recalls values at 0.85 but structures at 0.58, and triplet F1 falls from 0.32 to 0.011 without its structure module (ie F9, data F23).

Three upgrades make the loop honest and useful:

1. **Budget in harness-metered training tokens** (B), plus a persistent eval server that hot-swaps LoRA adapters. Throughput tricks and node contention cannot buy score, and three trainer GPUs run instead of one.
2. **Statistical honesty is measured, not asserted:** placebo arms, an eval-repeatability floor, score.py's family×seed hierarchical bootstrap, a fresh-item CONFIRM tier, a SEALED lockbox, and a Hyperband baseline matched on GPU-hours (C and judges).
3. **The product is records.** The SAR-table stage starts in week 5 against a hybrid baseline: CWU MOL files, OPSIN, a permissive specialist OCSR, zero-shot 8B linking. Production stays hybrid whatever the loop finds (ie F31).

Populations, lanes and LLM critics stay off until a measured failure demands them. In Agora the first 3 improvements gave 93% of the descent (eco F12); lineage, not agent count, drove valid improvements (16 vs 3; eco F11).

## 2. Staged scope and exit criteria

| Phase | Weeks | Inner loop optimizes | Scalar |
|---|---|---|---|
| 0: infra + calibration | 1-2 | — (harness, W0 warm start, noise measurement) | — |
| 1: S1 crops | 2-4 | One crop → structure (stereo, abbreviations, salts, large molecules); Markush diagnostic only | `val_struct` (§4.1) |
| 2: S2 SAR regions | 5-7 | Table crop + scaffold crop + footnotes + caption → compounds + records. R-group rows go to the harness enumerator (ie F23); names go to OPSIN | Composite, macro over patent/paper/SI; S1 as veto slice |
| 3: documents (deferred) | 8+ | Paper+SI and patents: "Example N" coreference, Markush claims, provenance, ADMET/PK keys, cross-document merge | Adds the page-level joint metric |

Why this order: 72.5% of activity values sit in tables (data F23); page eval costs ~1 s/page even below 2B (vlm F10); Markush scoring is not yet trustworthy (see (c)).

**Phase 0 exit (end of week 2):** score.py v1.5 S1 items, golden pairs and `formats.py` round-trips green; 20 runner trials with valid receipts; same-adapter re-scores within 0.1 pp (else deterministic decoding, §5); σ_seed (5 seeds), SE_diff, design effect and throughput measured; `TOKEN_BUDGET` and `keep_rule.yaml` frozen by pre-registered rules; ≤20% false-PROMISING over 20 placebos; W0-2B ≥ zero-shot + 20 pp on DEV.

**Phase 1 gate (Fri Nov 13).** All of:
1. ≥3 ESTABLISHED keeps, or a plateau (none in 3 nights with ≥150 candidates).
2. The ρ-audit passes (§5).
3. One SEALED run: the promoted 8B beats W0-8B (paired 95% LB > 0).
4. Placebo ESTABLISHED rate ≤2% over ≥30 placebos.
5. Agent-vs-Hyperband result recorded.
6. S2 ready: key v2, enumerator tests, ≥100 DEV tables (140 by week 5).

**Pivot.** If the 8B trails the best Track-C specialist by >5 pp on the SEALED real-literature slice and the frontier gains <1 pp/week for two weeks, the specialist becomes the production structure tool and the VLM loop moves to S2 linking. The specialist is OCSRGlyph or MolScribe re-scored by score.py; thresholds are never anchored to published numbers from other scorers (ocsr F13, met F3).

**Phase 2 exit (≈ week 7).** On CONFIRM:
- macro record F1 ≥ hybrid + 10 pp (LB > 0);
- record precision ≥0.85;
- `id_link_f1` ≥0.85;
- no S1 veto fires;
- ≥40% curator-time saving (§4.5).

## 3. File layout and the fixed/editable contract

```
research/drug-vlm/
  program.md   (human; agent read-only)     train.py  (THE ONLY AGENT-EDITABLE FILE)
  run.sh       ./run.sh "hypothesis" [--wave 3] | --smoke | --status
  analysis.ipynb
  harness/     read-only, deployed as /opt/dvlm-harness@<sha>
    prepare.py   constants (STAGE, TOKEN_BUDGET, caps, seeds); builds pools, tiers, W0
    dvlm/        load_base(), pools(), Meter, render(), augment ops, rewards, save_adapter()
    formats.py   registry -> RDKit mol: smiles, cxsmiles, molfile, graph_json_v1,
                 rgroup_table_v1, records_yaml_v1, iupac_name (OPSIN)
    abbrev_v1.tsv targets_v1.tsv units_v1.tsv annotation_guide.md enumerate.py
    score.py test_score.py golden_pairs.tsv evalserver.py keep_rule.py keep_rule.yaml runner.py
  results.tsv lineage.jsonl failed_descriptors.tsv   (runner-written; agent read-only)
/srv/dvlm/pools (ro)  /srv/dvlm/w0  /srv/dvlm/eval/{viz,dev,confirm,reserve,sealed}/{images,gold}
```

**Decision: one editable file (A), not B's `recipe.py` plus a KNOBS block.** B's worries were crashes and data comparisons confounded by hyperparameter changes. Here, `train.py` (~400 lines by default) sits on a tested read-only library; a smoke rung catches crashes; and the §6 statistics handle confounding. Capping HP edits would not stop HP tuning (eco F28), and LR is the highest-value knob (vlm F28).

**Contract.** Inputs come only through `dvlm` (W0+PEFT, metered pools, gated renderer, augmentation ops, rewards reusing score.py normalization); the seed comes from `DVLM_SEED`. Outputs:
- `/out/adapter`: a LoRA (rank ≤128) on LLM linear layers takes the fast path; ViT blocks or full weights take the allowed slow path (merged, fresh vLLM load, +4 min).
- `/out/infer_config.json`: prompt ≤2,000 characters, `output_format` and grammar id from the registry, `max_pixels` ≤1024², greedy or vote k≤4.

Everything else (mixture, curriculum, augmentation, losses including MRT, LoRA targets, LRs, schedules, EMA) is fair game.

**Mechanical enforcement** (eco F4, F10, F29; met F29):
1. Harness, `program.md` and pools are root-owned, `0444`, `chattr +i`; the runner executes only `@sha` copies.
2. **Diff gate:** a commit touching anything but `train.py` → `gate_fail`.
3. **Static scan:** ≤1,500 lines; no network/subprocess/ctypes imports, `from_pretrained`, weight or eval paths, literal seeds, or meter/hook tampering; no ≥15-character string hitting a Bloom filter of eval SMILES, InChIKeys or document IDs.
4. **Training container:** `--network none`, no API keys, read-only pools, cpuset/RAM cgroups, hard kill at the cap.
5. **Receipt:** the adapter loads on the W0 sha; meter tokens agree with optimizer steps (±2%); ‖ΔW‖ > 0; the runner times the run and records SHA-256 of `train.py`, pools, W0, adapter, harness and gold.
6. **Data gate:** every render passes through `dvlm.render`; molecules whose eval InChIKey block 1 or SAR-series Murcko scaffold is blocklisted are silently replaced, and the agent sees only a bucket (0 / <0.5% / ≥0.5%), so the gate is no membership oracle (B).
7. **Separate eval and scoring:** the eval server (UID `evalsvc`, images only) decodes; a CPU scorer (UID `scorer`, the only gold mount) emits signed JSON. Scores never come from agent stdout.

## 4. The metric

### 4.1 Primary scalar (S1)

`val_struct = Σ_s w_s · struct_f1_s` over a source partition:

| Source | Weight |
|---|---|
| In-house | 0.30 |
| BioVista-OCSR | 0.30 |
| USPTO 2025-26 | 0.20 |
| MolRecBench-Wild (MRBW) | 0.20 |

Weights are renormalized until in-house data arrives in week 3. They mirror the team's documents rather than MRBW's general-chemistry journals (a rigor-judge point). A partition avoids C's double counting.

On one-structure crops, `struct_f1` is exact match with a precision penalty. Invalid, unconvertible, unknown-abbreviation and missing outputs are wrong; extras are false positives; the first prediction wins.

**Representation independence, as with val_bpb.**
- `formats.py` converts every registered format to an RDKit molecule. Superatoms expand via the frozen `abbrev_v1.tsv`; an unknown abbreviation counts as wrong, never `*` (met F2).
- score.py keys the molecule: the stereo-aware InChIKey of the FragmentParent+Uncharger parent, or CXSMILES keys for enhanced stereo and dummy atoms. The output language cancels like vocabulary in bits-per-byte.
- Unscorable gold is excluded on both sides and coverage is reported.
- Stereo is strict: no "adaptive" rule (the ocsr F13 convention hides invented stereo). Tanimoto never decides (met F5, F10).

### 4.2 Calibrated vetoes and alarms

A guardrail vetoes only if Δ < −max(floor, 2.5·SE_bootstrap):

| Guardrail | Floor |
|---|---|
| Invalid-structure rate | 0.3 pp |
| Multi-fragment rate | 0.3 pp |
| Each hard slice (≥300 items): chiral, abbreviation, >50 heavy atoms, scanned/low-resolution, in-house | 1.5 pp |
| 300-item in-house table-cell transcription probe (guards S2 skills; vlm F25) | 2 points |

Fixed vetoes, without an SE term: decode cap exceeded or parse failures on >1% of items; projected 8B cost >0.5 GPU-s/crop.

Eight one-sided tests at z = 2.5 give ≈5% family-wise false vetoes. Placebos measure the real rate, which must stay ≤10%.

**Alarms only, never vetoes:** canary gap (100 never-published molecules) and perturbation gap (300 re-encoded crops).

**Diagnostics:** no-stereo F1; Tanimoto; `id_link_f1`; per-source F1; salt form (after the patch); Markush on IP5-M/M2S/USPTO-M; the Keylab suite; teacher-forced NLL within one format (replaces C's MI8, which met F26 does not support); ECE of a harness-computed confidence (mean token log-prob; ocsr F2).

### 4.3 S1 evaluation tiers

All splits are by article or DOCDB family.

| Source | viz | DEV | CONFIRM (+reserve) | SEALED | License / note |
|---|---|---|---|---|---|
| MRBW 2026-08-19, SMILES-evaluable (2,392; subset C has only 298, ie F37) | 100 | 1,200 | 700 (+300) | — | No card license; CC-BY source papers. Eval-only after legal sign-off |
| BioVista-OCSR non-Markush (≈half of 8,861) | 100 | 1,000 | 600 (+400) | — | MIT. Drawn from the test split, so we claim no public BioVista numbers; the ≈886-item val split is too small |
| USPTO 2025-26 CWU crops, MOL-derived, chemist-checked | 50 | 600 | 400 (+200) | — | Public domain; check 37 CFR 1.71 (data F37 [U]) |
| In-house papers/SI/patents (week 3) | 50 | 600 | 400 (+200) | 1,000 | Proprietary. SEALED = documents from ≥2026-07-01, EPO/CNIPA/JPO, project reports |
| Keylab CLEF/JPO/UOB/ACS/USPTO-10K; IP5-M/M2S/USPTO-M | nightly diagnostics | | | | Mixed / CC BY 4.0 |

**Power.** With n = 3,400 in ~500 clusters, discordance ≈6% and design effect ≈1.3, SE_diff ≈ 0.48 pp. With σ_seed ≈ 0.5 pp, SE_tot is ≈0.75 pp (1 seed vs 3 banked) and ≈0.63 pp (3 vs 3). The 80%-power MDE is ≈1.6 pp (met F22). A marginal 12-minute stage probably moves 0.5-2 pp, so expect few keeps: the honest price.

### 4.4 S2 metric and record schema v1

**Scalar:** 0.5·struct_F1 + 0.5·record_F1 per document type, macro over patent/paper/SI (B).

**Key v2:** structure, endpoint (closed vocabulary), target (UniProt or organism via `targets_v1.tsv`), variant, assay_type, species, cell_line/matrix, condition, relation, unit class, value.

**Gold-defined fields rule.** A field counts only if gold fills it. Inferring the likely species is rewarded; random filling is not. This closes v1.4's verified holes: rat vs human clearance and biochemical vs cellular IC50 currently share a key.

**Value matching** keeps v1.4 rules: relative tolerance 1e-3 at the same scale; converted values must land inside the gold's printed rounding interval.

**Captured now, scored in Phase 3:** n, ±SD, replicate index; `stereo_status` (absolute / relative / racemic / arbitrary single enantiomer = CXSMILES `o` group); ADMET/PK fields (dose, route, formulation, pH, kinetic vs thermodynamic solubility, Caco-2 direction and efflux ratio, CYP isoform, time point); provenance (page, bbox, table, row, column, footnote ids).

**Binned patent activity:** gold stores the printed label ("A", "+++") and the legend-resolved interval as alternates.

**Scope rules (`annotation_guide.md`):** a record is one printed value per compound × endpoint × system × condition. Reference compounds are tagged `role=reference`; intermediates count only with data; cited values are kept with `cited=true`; paper/SI duplicates are scored per document; `compound_id` is the label as printed ("Example 12, Step 3"); stereo and E/Z are scored as drawn.

**Sizes:** DEV 140 tables (~70 documents: 80 in-house, 40 BioVista-val with PDFs fetched by PDB ID, 20 ChemX MIC-vs-organism tables, MIT, µg/mL converted via MW); CONFIRM 100; SEALED 60 (EPO/CNIPA, post-cutoff). Errors cluster by table, so the effective n is close to the table count: SE ≈2-3 pp, MDE ≈6 pp, and S2 hunts large levers. DEV grows ~30 tables/week; the ICC is measured in week 5, not borrowed (met F22's 2.9 is hypothetical).

### 4.5 Validity, label quality, contamination

1. **Downstream validity** (S2 gate, 30 held-out documents with ChEMBL/BindingDB coverage):
   - database-record recall, compared in log10 within ±0.01; recall only, because the databases curate subsets and normalize values (ie F30);
   - curator correction time vs manual entry with pre-fill (target ≥40% saving; grounded output halved MolScribe review time, ocsr F2);
   - rank agreement between record F1 and curator time across ≥4 models;
   - F0.5 reported for SAR modelling.
2. **IAA.** 20% of gold is double-annotated; record-level IAA F1 is the ceiling. Weekly disagreement mining leads to human-only fixes, versioned gold and a re-baseline (met F21).
3. **Abbreviation check:** `abbrev_v1.tsv` must reproduce every superatom-containing gold structure.
4. **Contamination probe.** Run zero-shot 2B/8B on MRBW by year and on layout-matched in-house documents before vs after 2026-07-01. A pre-cutoff advantage >2 SE flags that source.

## 5. Budget per experiment

**Throughput.** A 512² crop is 256 vision tokens (vlm F32); with prompt and answer, T ≈ 436. A 2B LoRA costs ≈6·N·T plus the ViT ≈ 6e12 FLOPs. At 10-20% MFU that is 16-33 crops/s (vlm F33 [U]); the plan uses 25/s, about 10.9k tokens/s.

**`TOKEN_BUDGET` = 8M tokens (~18k crops), counted by `dvlm.Meter`, with a hard kill at 1,500 s.** Under a token budget, verbose formats and higher resolution pay their own way, and contention costs wall-clock rather than samples (B).

The budget is frozen by a pre-registered rule:
- Run a grid of {4M, 8M, 16M} × 3 seeds plus 4 variants.
- Choose the smallest budget with σ_seed ≤0.7 pp whose ranking matches 16M (Kendall τ ≥0.67).
- If the median training time at 8M exceeds 15 min, use 6M.

| Step | Where | Seconds |
|---|---|---|
| Container + load merged W0-2B | trainer | 45 |
| Train 8M tokens | trainer | 735 (555-1,140) |
| Save adapter + receipt | trainer | 15 |
| Hot-swap LoRA; decode 4,100 items (DEV + probe + canary + perturbed), LoRA ≈1.35× slower (vlm F31) | eval GPU | ≈100 |
| NLL on 1,000 crops | eval GPU | 15 |
| Score + hierarchical compare | CPU | 40 |

That gives ≈13.3 trainer-minutes per run, **4.5 runs/h/GPU**, and ≈16 min to a decision.

**Overnight.** A 14-hour night on 3 trainer GPUs is ≈190 runs. After 10% placebos, re-baselines and ≈0.5 replicate runs per candidate, that is ≈110 candidates. The agent (waves of ≤3) is likely the bottleneck: expect 60-110. On **1×H100**, expect ≈63 runs (≈40 candidates), with eval sharing the GPU.

**Eval server.** vLLM serves W0-2B with LoRA enabled (4 slots, rank ≤128); verify Qwen3-VL LoRA support in week 1 (fallback: merged weights, +3-4 min per trial). Batched greedy decoding is not batch-invariant, so fix request order and batch size, use batch-invariant kernels if available, and add the measured floor to SE_tot. `EVAL_DECODE_CAP` = 600 s per DEV pass (≈6× baseline), enough for traversal CoT (~500 tokens; ocsr F9) or vote@4; undecoded items score wrong, and GPU 4 adds a second server when the queue exceeds 2. **Deployment target:** projected 8B cost ≤0.5 GPU-s per crop (≈$0.0004; vlm F11) and ≤5 GPU-s per page.

**GPUs (Phase 1):** 0-2 inner loop; 3 eval server; 4 placebos, second eval server and re-baselines; 5-7 W0 builds, then R2, ρ-audit, Hyperband, and data-engine teachers by day. CPU: 24-core cpusets per trainer slot and 16 for eval/scoring; cgroup RAM caps; shards on NVMe.

**Rungs:** R0 60-s smoke → R1 screen (1 seed) → R1 replicate (+2 seeds) → CONFIRM (eval only) → R2 (8B LoRA on W0-8B, 32M tokens ≈2.6 h at ~8 crops/s [U], 2 seeds, weekly) → R3 release at gates.

**W0 (the tokenizer analogue, pinned per stage).** A Qwen3-VL-2B LoRA (r=64, 300k samples, ≈3.4 GPU-h), then selective full fine-tuning on 1.2M samples with the ViT frozen and split LRs, since direct full SFT diverged in MolSeek (ocsr F16). Cost ≈9.4e18 FLOPs ≈18 GPU-h, at ×1.3 the LoRA FLOP model for weight gradients, preceded by a 4-point LR sweep (≈12 GPU-h). Targets are multi-format by prompt tag and superatom-faithful. W0-8B is a LoRA r=128 on 1.2M samples, ≈12 h on 4 GPUs.

**ρ-audit with power and a reliability ceiling.** 24 recipes spread over the R1 Δ range (6 top / 12 middle / 6 bottom; C) run at R2 (≈62 GPU-h); 8 are re-seeded at both rungs (≈22 GPU-h) to give within-rung reliabilities r1, r2. **Pass:** Fisher-z 95% LB of ρ ≥0.3 (ρ=0.6 at n=24 gives CI ≈[0.26, 0.81]), disattenuated ρ/√(r1·r2) ≥0.6, and ≥75% of ESTABLISHED keeps positive at R2. If the CI straddles 0.3, add 12 recipes and let at most 2 unconfirmed keeps stack meanwhile. **Fail** (disattenuated ρ <0.4): move to a 16M budget or the 4B proxy; a second fail sends search to Hyperband at R2. LR, schedule and resolution edits are allowed at R1 but tagged horizon-sensitive (eco F42) and need R2 before stacking.

**Matched non-agent baseline (week 4).** Optuna-TPE with ASHA (2M → 8M → 32M tokens) over §7's numeric knobs, same W0 and keep rule, 84 GPU-h. That equals the agent's first two nights (eco F36, F40). The metric is confirmed gain per GPU-hour.

**Crash allowance.** The first 3 nights reserve 20% of slots for crashes and timeouts (Amazon lost 11 of 30 iterations to evaluator timeouts; eco F15). Infra failures retry once and are logged `infra`. A cheap-model Code Fixer switches on above a 15% nightly crash rate.

**Cost model (6 weeks):**
- GPU: 8,064 GPU-h ≈ $21.7k at $2.69/h (vlm F11).
- Agent API: runner-capped at $60/night, ≤$2k.
- Engineering: ≈18 person-weeks.
- Chemists: ≈400 h (S1 crops ≈45 h, 300 S2 tables ≈305 h, Markush ≈30 h, adjudication ≈20 h); ≈310 h fall in weeks 1-6 and the SEALED S2 tables slip to week 7.
- Legal: ≈20 h.
- Lead times: USPTO bulk CWU is multi-TB with an unverified layout (data F1 [U]), 2-3 weeks; BioVista, ChemX and DocSAR ship IDs/DOIs, so PDF fetching takes 1-2 weeks.

## 6. Keep/discard under noise

(b) gives the exact steps. Rationale:

- **Every comparison uses score.py's hierarchical family×seed bootstrap**, so SE_tot² = SE_diff² + σ²(1/k_c + 1/k_i) + floor². The old seed-pooled test gave 15-22% false keeps at a nominal 5%.
- **Common random numbers:** the runner owns seeds, so the data order is shared when the data stream is unchanged.
- **Winner's curse.** The best of ~100 null screens sits about 2.5 SE ≈ 1.9 pp above zero (met F24), so a screen never advances the branch. Seeds share DEV items and cannot remove the item-level curse (C). Hence **CONFIRM on fresh items**: one bit, ≤10 queries/day, and 30% of items rotated in from the reserve after 100 queries or monthly (met F25 [U]).
- **Morning re-baseline** replaces the banked files of a lucky incumbent (A).
- **Empirical-Bayes shrinkage** applies to the Δs that are displayed and ranked (C).
- **Placebos (C)** fill 1 in 10 screen slots, unannounced. Targets: PROMISING ≤20%, DEV_KEEP ≤5%, ESTABLISHED ≤2%, vetoes ≤10%. False-keep rates are measured, not multiplied out as in A's "<1%".
- **Tiers:** dev-viz per item; DEV rounded (0.1 pp, slices 0.5 pp); CONFIRM one bit; SEALED a human-only lockbox at gates and final release (≤4 uses in 8 weeks). Weekly choices use CONFIRM and R2 (B's lockbox, fixing A's and C's reuse).
- **Final selection:** the top-3 ESTABLISHED states at R2 go to CONFIRM; only the winner goes to SEALED (eco F16).
- **Simplifications** need CONFIRM non-inferiority (LB > −0.5 pp), at most one per day. A weekly drift check reverts an accumulated slide.

## 7. The agent's search space (program.md priors)

Published gains are mostly from-scratch effects; on top of W0 expect less. **H** = horizon-sensitive, needs R2 before stacking.

| # | Edit | Evidence | Marginal prior |
|---|---|---|---|
| 1 | Per-module LR; schedule derived from the budget | LRs span 3 orders of magnitude (vlm F28) | ±1-3 pp, H |
| 2 | Real-crop share 10/30/50% | data F3; ACS 0.15→0.46 [U] (data F7) | +1-4 |
| 3 | Augmentation strength + weak→strong curriculum | 40.1→69.5→76.9 (data F6) | +0.5-2 |
| 4 | Superatom-faithful labels (harness expands) | <20%→70%+ (ocsr F29); GTR's +14 is the whole system (ie F18) | +1-3 |
| 5 | Chiral ×2, wedge↔wavy swaps, decorative-wedge negatives | ocsr F26, F30 | +1-3 on chiral slice |
| 6 | LoRA targets (attention / gate+up / all), rank 16-128 | +24.9/−0.6 learning/forgetting (vlm F25) | ±1-2 |
| 7 | `max_pixels` 384²-768²; aspect padding | vlm F32, ocsr F31 | ±1-3, H |
| 8 | EMA / checkpoint averaging | ≈63% of speedrun gain (eco F8); vlm F12 | +0.3-1 |
| 9 | Hard-example weights from per-stage precomputed features (W0 loss, ensemble agreement 0.6-0.9) | data F5, ie F32 | +0.5-2 |
| 10 | 5-30% general document/table replay | eco F9, vlm F26 | Holds probe |
| 11 | SMILES / CXSMILES / `graph_json_v1` with coordinates | ocsr F27, F28, F2 | ±2; may help stereo |
| 12 | Graph-traversal CoT | +2.3 to +3.1 (ocsr F9) | +1-2 |
| 13 | Batch size vs steps | eco F2 | ±1 |
| 14 | Grammar-constrained decoding | vlm F14 | Fewer invalid outputs |
| 15 | Vote by InChIKey, k≤4 | MV@8 +1-4 (ocsr F26) | +0.5-2 |
| 16 | Tiling for >50 heavy atoms | ocsr F1, F31 | +3-8 on slice |
| 17 | Randomized SMILES targets | ocsr F27 | ±1 |
| 18 | COMO MRT: R = 0.1·valid + 0.5·sim·valid + 0.4·exact | +12-16 on JPO/ACS (ocsr F8) | +1-5, RL lane |
| 19 | GRPO/DAPO with strict InChIKey reward and an invented-stereo penalty | ChemVLR's reward was Tanimoto=1, which is stereo-blind (ocsr F20, met F5); GSPO hurt EM (met F10) | 0 to +3, RL lane |
| 20 | Top-k ViT blocks at 0.1× LR (slow path) | vlm F25; data F7 [U] | Low |
| 21 | Selective full-FT phase (slow path) | ocsr F16 | Unstable |
| 22 | Answer-only loss; label smoothing | — | ≈0 |
| S2 | YAML records (vlm F13); IdtVP (ie F13); `rgroup_table_v1` + enumerator (ie F23); synthetic SAR tables on real layouts (data F10) | — | Large |

**Order.** Nights 1-4: items 1-10. Week 4: 11-17 plus the RL lane (18-19). The RL lane gets a fixed 1,800 GPU-s budget and an automatic equal-GPU-seconds SFT control, because sampling compute escapes the token meter (C).

**Known negatives:** direct full SFT diverged (ocsr F16); Thinking variants are worse at OCR (vlm F2); RL without SFT is useless (ocsr F20); generic synthetic layouts hurt (data F10); Liger lacks Qwen3-VL (vlm F31).

## 8. Data strategy

**Fixed pools.** Each sample carries license, track, source and family.
- **P-render:** 2M PubChem/ChEMBL SMILES, minus eval block-1 InChIKeys and SAR-series scaffolds, rendered by RDKit, Indigo and CDK in RanDepict styles (data F31). Two label views: superatom-faithful and expanded.
- **P-uspto:** MolScribe's 680K pairs (HF tag MIT, data F3), cleaned (`*` remapped, CDX over MOL, ensemble-disagreement drops; data F2) and stripped of eval families and classic-benchmark patents (data F30); our own 2001-2024 CWU mining arrives as v2 in week 4.
- **P-stereo:** Stereo-200k (Apache).
- **P-markush:** MarkushGrapher-1 data (CC BY 4.0) and USPTO-MOL-M (54,785; its card is untagged, so it needs legal review; ocsr F38).
- **P-inhouse** and **P-replay** (license-checked).
- **P-sar:** synthetic SAR tables on perturbed real layouts with ChEMBL/BindingDB values (data F10, F38), ChemX and in-house training tables. BindingDB/ChEMBL are never full-table labels: they are incomplete and normalized, and would teach the model to skip records. They only pre-fill annotation and measure recall.

**Label fidelity (B, harness policy):** superatoms as drawn; `*` only for drawn wildcards; USPTO `*` remapped to the visible label; CDX over MOL; stereo labelled only where drawn; counterions kept as drawn and scored on the parent, with salt form as a diagnostic (fixes B's lever 18).

**Leakage controls:** family/article splits; InChIKey block-1 dedup across pools and tiers; Murcko series kept in one tier and blocklisted from renders; pHash dedup; time-split SEALED; the data gate on every materialized sample; a nightly overlap audit of logged sample IDs.

**License tracks.** Track R (separate leaderboard, never shipped): MolParser-7M, MolGallery, MolParser-Mobile, MolDetv2, COMO weights, MolMole_Patent300, and **MarkushGrapher-2 and MarkushGlyph despite their Apache weights**, since both trained on CC BY-NC-SA MolParser rows (ocsr F25, F26). glyph-datasets stays in Track R until its MolParser rows are removed. Any NC row in a commercial run invalidates that run and its descendants (C). OCSRGlyph enters Track C after provenance review. In-house subscription PDFs are eval/inference-only unless licensed for training (data F35 [U]). Frontier-API teachers need legal sign-off on output terms.

**Production routes beyond the VLM** (in the week-3 hybrid; scored in Phase 3):
1. US-patent CWU MOL/CDX files are read directly and cross-checked by VLM OCSR, since connection tables can be wrong (data F2); the VLM-only eval track strips CWUs.
2. For examples defined by name, the model transcribes the name and OPSIN (MIT) converts it; disagreement with OCSR is flagged.
3. Value QC: CAPRICHO-style 3.0/6.0 log-unit flags and plausibility ranges (ie F30).
4. Our own detector, bootstrapped from Docling's MIT classifier (MolDetv2 is NC). Detection is unsaturated on drug papers (BioVista AP-small 0.050; ie F33), so a nightly page-level joint report (IoU ≥0.5 AND key; met F17) runs on 50 in-house pages from week 6.

**Data engine** (offline, human-gated, weekly bump then re-baseline): harvest Track-C pages; sample from the 0.6-0.9 ensemble-agreement band; use ChemScraper vector labels for born-digital PDFs. Chemists correct at ~30 s/molecule, so double review yields ≈2.4k crops per 40 h, not B's 5k.

**Fixed:** pools, tiers, generators, tables, W0, caps, budget.
**Agent-tunable:** weights, feature filters, augmentation code, render specs, curriculum, label view, losses, format.

## 9. Base model

- **Inner loop: Qwen3-VL-2B-Instruct.** Apache-2.0; OCRBench 858 vs 792 for Thinking; 32 px per token; supported by TRL, ms-swift, LLaMA-Factory and vLLM (vlm F1, F2, F29).
- **R2 and production: Qwen3-VL-8B-Instruct** (OCRBench 896; an 8B LoRA fits one H100, vlm F30). It shares the tower, tokenizer and template with the 2B, which makes transfer most plausible.
- **Fallback proxy:** Qwen3-VL-4B (881).
- **Week-1 bake-off** (zero-shot plus 2 h of identical SFT). Switch only for a gap of ≥3 pp, to InternVL3.5-2B/4B (Apache; 836/822) or Gemma 4 E4B/12B (Apache; 70-1,120 token budgets; vlm F7 corrected).
- **Excluded:**
  - Qwen3.5: thinks by default, needs flash-linear-attention, and its DeltaNet state leaks across packed rows (vlm F4).
  - Gemma 3: fixed 896 px.
  - AGPL/NC models (vlm F35).
  - Chemistry VLMs: ChemVLM 8.03 and ChemDFM-X 19.06 on revised MRBW (ocsr F10).

## 10. Organization

- **One agent, one branch, three trainer slots,** filled by waves of ≤3 single-factor candidates (SkyPilot's agent did this unprompted; eco F6). The deterministic runner is the statistician.
- **Lineage:** runner-written `lineage.jsonl` (hypothesis, descriptor, diff, parent, seeds, metrics, data card, receipt, status, failure reason). The agent sees the incumbent, its last 15 rows and the tried-and-failed table; `run.sh` rejects failed descriptors unless the hypothesis says what changed (Amazon re-tested failures 6+ times despite prose; eco F15).
- **Human:** daily 15-20 min review (frontier, placebos, ρ, transcript flags), plus a 2-5-sentence program.md redirect when the stagnation ping fires (2 nights or 150 candidates without ESTABLISHED). Agents change strategy in 2.1% of transitions, and one redirect was worth up to 17 points (eco F30). Weekly: R2 and pool bumps. Gates: SEALED and the decision.
- **Critic:** weekly transcript sweep by another model family, flags only (met F35).
- **Scale only if forced** (stagnation ping twice in a week, after week 4): a second agent with a different focus. C's MAP-Elites, Director and lanes are deferred.

## 11. Anti-reward-hacking and integrity

Beyond §3: runner-owned seeds with 3 seeds plus CONFIRM (Claudini fell back to seed search once stalled; eco ADDED); rounded feedback, a separate dev-viz set and a budgeted one-bit CONFIRM; schema-checked `infer_config` with its prompt scanned for SMILES; strict scoring (invalid, duplicate and missing all count against the model; no best-of-N; unknown abbreviations wrong; met F15, F16); canary and perturbation alarms; a nightly overlap audit; receipts binding code, data, weights and gold; a transcript sweep. On a confirmed hack: freeze, patch, re-baseline, mark the row `hack`.

## 12. Top risks, mitigations, pivot or abandon triggers

| Risk | Mitigation | Trigger → action |
|---|---|---|
| Proxy does not transfer (eco F42; no chemistry-VLM study) | Powered ρ-audit, horizon tags | Disattenuated ρ <0.4 twice → Hyperband at R2 |
| Effects below the MDE | 5k DEV, CRN, re-baseline | 0 ESTABLISHED in 2 weeks with placebos in spec → compute goes to the data engine and a W0 rebuild |
| Agent no better than Hyperband | Week-4 matched baseline | Agent ≤ Hyperband → knobs go to Hyperband; the agent keeps code-level levers |
| VLM OCSR ceiling (ocsr F16) | Real data, superatoms, graph_json, MRT | §2 hybrid pivot |
| CONFIRM overfitting | Budget, rotation, lockbox | Pass rate <30% → rotate, raise z |
| S2 annotation bottleneck | Pre-fill; public tables first | <140 tables by week 5 → start on public tables; exit slips |
| License contamination | Manifests, tracks | NC row → invalidate the run and its descendants |
| Infra fragility | Smoke, retries, reserve | Crash rate >15% → Code Fixer |
| Eval nondeterminism | Measured floor | Floor >0.2 pp → deterministic decoding |
| Base-model contamination | Probe, time-split SEALED | Flag the source |

**Abandon** the autonomous loop by week 6 if no frontier beats the hybrid on CONFIRM and Hyperband matches the agent. The harness, data engine and hybrid pipeline remain the product.

## 13. Weeks 1-6

| Week | Work | Deliverables |
|---|---|---|
| 1 (Oct 19-23) | Containers, runner, `run.sh`, eval server. score.py v1.5 S1 items, `formats.py`, golden pairs. Public downloads; P-render; DEV v1 (2,800 crops). Legal tickets. Baselines re-scored. Bake-off. W0 LR sweep (Thu-Fri). USPTO bulk download and in-house annotation start | Harness and tests; baseline rows |
| 2 (Oct 26-30) | W0-2B ready Monday. Calibration (budget grid, σ, repeatability floor, 20 placebos, ICC). Freeze keep_rule; program.md v1. **First autonomous night Thu Oct 29.** W0-8B over the weekend | Calibration report; ≥100 rows |
| 3 (Nov 2-6) | DEV v2 (+600 in-house), re-baseline. Nightly loop. ρ-audit runs. S2 guide, key v2, enumerator, table annotation. Hybrid pipeline v0 | ρ report; hybrid v0 |
| 4 (Nov 9-13) | Hyperband (84 GPU-h). RL lane with controls. R2 promotion. **S1 gate Friday** (SEALED #1, pivot decision). W1 build over the weekend | Gate memo; OCSR-v1 8B |
| 5 (Nov 16-20) | S2 calibration; S2 nights from Wednesday. S1 continues on 1 GPU. 300 Markush items annotated | S2 vs hybrid |
| 6 (Nov 23-27) | S2 nights. Validity study (30 documents). Page-level report. Cost retrospective. Phase-3 plan | Extractor-v1 report; S3 plan |

---

## (a) results.tsv column spec

The runner writes it (tab-separated, untracked); the agent may read it.

| Column(s) | Meaning |
|---|---|
| `commit`, `parent` | Short SHAs |
| `stage` | S1 / S2 |
| `rung` | R0 / R1s / R1r / CONF / R2 / R3 |
| `kind` | cand, rebase, base, hyperband |
| `seeds` | Seeds used |
| `status` | gate_fail, crash, infra, invalid, veto:<name>, discard, promising, dev_keep, confirm_fail, established, simplify_keep, r2_revert, hack |
| `score` | Mean over seeds, 0.1 pp |
| `delta` | Empirical-Bayes shrunk, 0.1 pp |
| `ci90_lo`, `ci90_hi`, `p_not_better` | Hierarchical bootstrap |
| `seeds_pos` | e.g. 2/3 |
| `confirm` | PASS / FAIL / - |
| `worst_slice`, `worst_slice_delta` | Delta at 0.5 pp |
| `invalid_rate`, `multifrag_rate`, `nostereo_f1`, `probe_delta`, `canary_gap`, `perturb_gap`, `nll` | Guardrails and diagnostics |
| `train_tokens_M`, `samples_seen`, `steps`, `train_s`, `eval_s` | Budget |
| `decode_tok_per_item`, `proj_cost_gpu_s`, `peak_vram_gb`, `gpu_h` | Cost |
| `real_share`, `stereo_frac`, `abbrev_frac`, `mean_tok` | Data card |
| `horizon_tag`, `descriptor`, `description` | Horizon flag, tried-and-failed key, free text |
| S2: `record_f1`, `record_p`, `record_r`, `id_link_f1`, `orphan_rate`, `s1_veto_delta` | Record metrics |

A human-only `results_full.tsv` adds:
- raw deltas;
- per-slice deltas with SE;
- the placebo flag;
- the six SHA-256s (`train.py`, W0, pools, adapter, harness, gold);
- RDKit, vLLM and OPSIN versions;
- the receipt id.

## (b) Keep/discard procedure (the runner executes; the agent reads)

1. **Gate:** diff touches only `train.py`, passes the static scan and Bloom filter, and has a new descriptor (or states "changed: …"); else `gate_fail`.
2. **R0 smoke** if loop, loss, model wrapping or `infer_config` changed; on a crash the agent gets the log tail and ≤2 fixes, then `crash`.
3. **Screen:** train seed s0 (shared with the incumbent's s0) under `TOKEN_BUDGET`; a bad receipt → `invalid`.
4. **Compare:** decode DEV plus probes; `score.py score`; `compare --base inc_s0,inc_s1,inc_s2 --new cand_s0`.
5. **Hard vetoes** → `veto:<name>`: decode cap missed on >1% of items; parse failures >1%; projected 8B cost >0.5 GPU-s/crop; any guardrail Δ < −max(floor, 2.5·SE).
6. **PROMISING** if shrunk Δ ≥ +0.5 pp and `p_not_better` ≤0.20; else `discard` (worktree reset, descriptor recorded as failed).
7. **Replicate** s1 and s2 in parallel; compare 3 vs 3. **DEV_KEEP** needs mean Δ ≥0.5 pp, `p_not_better` ≤0.05, ≥2/3 seed-paired Δs positive, and step 5 still clean.
8. **CONFIRM** (≤10/day) on fresh items: PASS needs mean Δ >0, `p_not_better` ≤0.10, and no slice down by > max(2 pp, 2.5·SE). The agent sees PASS/FAIL only; FAIL → `confirm_fail` (dev-only in lineage).
9. **Horizon check:** if horizon-tagged, or 2 unconfirmed keeps already stack, run R2 (8B, 2 seeds) first; LB <0 → `r2_revert`.
10. **ESTABLISHED:** advance `autoresearch/<tag>`, bank the 3 per-doc files, and re-screen still-PROMISING wave-mates on the new incumbent.
11. **Simplification:** a ≥20-net-line deletion with mean Δ ≥ −0.2 pp skips step 6's margin; it needs a CONFIRM LB > −0.5 pp, ≤1 per day, plus the weekly drift check.
12. **07:00 re-baseline:** 3 fresh incumbent seeds; if their mean < banked − 1·SE_tot, replace the banked files (`rebase`).
13. **Placebos:** every 10th screen slot runs steps 3-8 unannounced; if a rolling-50 rate exceeds its §6 target, z += 0.25 and the human is paged.
14. **Logging:** only the runner writes `results.tsv` and `lineage.jsonl`.

## (c) Gaps and changes needed in score.py (v1.4 → v1.5)

Behaviours marked "verified" were reproduced today with RDKit 2026.09.1.

**S1 (week 1):**
1. `--slice-weights` with a slice-stratified bootstrap (today: micro-average; slices only printed).
2. `compare` outputs per-guardrail and per-slice paired Δs with SEs, for 2.5·SE vetoes (today: scalar only).
3. Emit gold and scorer hashes plus RDKit/OPSIN versions.
4. `golden_pairs.tsv` (~200: tautomers, salts, E/Z, enhanced stereo, dummy atoms, isotopes, metals) as a regression test before any toolkit bump.
5. Abbreviation-dictionary consistency test against every gold superatom.

**S2 (week 3):**

6. Record key v2 (assay_type, species, cell_line/matrix, variant), a UniProt/organism target map, and the gold-defined-fields rule. Verified: rat vs human clearance match; ErbB1 ≠ EGFR.
7. Value strings (verified): "12 ± 3" → categorical `cat:123`, silently wrong (parse mean/sd/n); "10 µM" in the value → `cat:10m` (parse number+unit); "1.2 × 10⁻⁵" unsupported; "IC50 (µM)" with an empty unit → class `unit:` (fall back to the header unit).
8. Binned activity: gold alternates (`value_alt`), label or legend interval.
9. Replicate fields (n, sd, index) as diagnostics.
10. Macro over document types; S1 kept as a veto slice.

**Phase 3 / diagnostics:**

11. Markush substituent-table F1; Sg compared only as canonical text today (verified); parsed-graph equality (ocsr F26) with no wall-clock MCS (met F13).
12. Salt-form diagnostic via the pinned ChEMBL Structure Pipeline (met F8); today Na acetate = acetic acid by design.
13. E/Z-only-agnostic diagnostic (the skeleton key strips all stereo), `stereo_status` scoring, an atropisomer flag (InChI blind spot).
14. Optional per-record `conf`: ECE and precision at 80/90% coverage.
15. Graph track for gold without SMILES (2,632 MRBW items), with coverage.
16. Page-level joint metric (`page`, `bbox`; TP = IoU ≥0.5 AND key; met F17) plus provenance accuracy.
17. Document the "≤"→"<" and "~"→"=" policies instead of changing them.

## (d) program.md outline

1. **Purpose and stage:** current STAGE, task unit, why structures first.
2. **Setup:** agree a tag; `./run.sh --check`; read in-scope files.
3. **What you can edit:** `train.py` only; the `dvlm` API; adapter and `infer_config` contract.
4. **What you cannot do:** edit harness/data/program; network, keys, new deps; other checkpoints; seeds; eval paths or literals.
5. **Budget:** tokens are the currency; caps and cost veto.
6. **Metric:** scalar, representation independence, strict stereo, vetoes, rounding.
7. **Running:** `./run.sh "hypothesis"`, `--wave`, `--smoke`; state expected Δ and a descriptor.
8. **Reading results:** columns; PROMISING / DEV_KEEP / ESTABLISHED; CONFIRM is one bit; the runner owns the branch.
9. **Priors:** the §7 table; horizon-tagged edits need R2.
10. **Known negatives:** plus "RL must beat its equal-compute SFT control".
11. **Simplicity:** deletions at equal score win; >60 added lines needs twice the margin.
12. **Tried-and-failed:** re-tests must state "changed: …".
13. **Crashes:** read the smoke tail; ≤2 fixes; infra failures are not yours.
14. **NEVER STOP:** keep 3 slots busy; when stuck, mine dev-viz errors and cited papers.
15. **Data and license policy:** Track C only; renders via the gate; no SMILES from memory in code.
16. **Integrity:** what is audited; placebos exist; hacks are frozen.
17. **Human redirects:** the latest directive overrides the priors.

## (e) Coverage of the judges' "missing_from_all" items and corrections

**Missing items:** §5 (eval server, repeatability, CPU/IO, crash allowance, cost target, Hyperband, cost model, ρ-audit); §4.2/§6 (placebos, false-veto control); §4.5 (validity, IAA, contamination); §4.1 (domain weights); §4.4/(c) (assay context, binned activity, ADMET/PK, replicates, stereo, scope rules, clustering); §8 (OPSIN/CWU, value QC, NC provenance, page metric, BindingDB incompleteness); §2 (re-scored thresholds).

**Corrections adopted:** detection unsaturated on drug papers (ie F33); 2.9 design effect hypothetical; 30 s/molecule per pass; MI8 dropped; ChemVLR reward stereo-blind; no adaptive stereo; Liger lacks Qwen3-VL; GTR +14 is whole-system; full-FT ×1.3 and staged W0; InternVL vision-LoRA claim is InternVL3-8B [U]; DataDecide is pretraining-only; MolSight throughput includes SVG; MarkushGrapher-2/MarkushGlyph are Track R; BioVista val split too small; MRBW subset C has 298 items; gaps planned against v1.4.
