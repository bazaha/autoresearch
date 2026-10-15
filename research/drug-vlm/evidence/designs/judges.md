# 评委打分原文（英文）

三个设计 agent 各自写出方案（`design_A.md` 最小忠实移植、`design_B.md` 数据中心、`design_C.md` 规模化研究组织 + RL），三个评委 agent 各从一个视角对照证据库打分（1–10），并实际运行了评分器。下表即 `../../README.md` 第 6.1 节的分数来源；最终综合方案见 `../design_synthesis.md`。生成日期：2026-10-15。

| 方案 | 可行性/算力经济 | 评测严谨性 | 化学领域正确性 | 合计 |
|---|---|---|---|---|
| A | 7.5 | 7.5 | 6.5 | 21.5 |
| B | 5.5 | 5 | 6 | 16.5 |
| C | 3.5 | 6.5 | 7 | 17.0 |

---

## Lens: feasibility（可行性/算力经济）

**Ranking:** A, B, C

### Design A — score 7.5

**Strengths**

- Fastest credible time-to-first-signal. One agent on one GPU, with the first autonomous night at the end of week 1 if S0 builds on Tuesday. The first SEALED eval and the 8B rank study come in week 2.
- The budget arithmetic is internally consistent and sits inside the digest's derived ranges. 6·N·T + ViT ≈ 6e12 FLOP/crop gives 16-33 crops/s at 10-20% MFU (vlm F33 [UNVERIFIED]). 1,220 s per run gives ~42 runs per 14-h night. S0-8B LoRA is ≈1.2M/(8 GPUs × 8/s) ≈ 5 h. All of these check out.
- Most compute-efficient funnel of the three. A 1-seed screen, with 2 more seeds only for PROMISING candidates, averages ~1.5 runs per candidate: ≈0.5 GPU-h, or ≈$1.4 at olmOCR's $2.69/h (vlm F11). CONFIRM is eval-only. This gives the lowest cost per ESTABLISHED keep.
- Day-4 calibration grid (TIME_BUDGET ∈ {300, 600, 900, 1800} × 3 seeds + 4 variants, ≈7-8 GPU-h) before the first night, with explicit rules for changing the budget. A throughput pivot applies below 10 crops/s.
- EVAL_WALLCLOCK_CAP, with undecoded items scored as wrong, makes inference-cost tricks (vote@k, long CoT) pay for themselves without new machinery.
- Smallest infrastructure surface: one editable file, a closed output-format registry, stage gates, and a week-3 hybrid-pivot trigger tied to a named baseline (8B below MolScribe's 62.3). Lowest infra-fragility risk.
- Annotation is costed and consistent: 1,600 double-reviewed crops ≈ 27 chemist-hours at ~30 s per molecule (data F5).

**Fatal or major flaws**

- The wall-clock TIME_BUDGET (900 s) becomes noisy once 2-3 trials share one node's CPU dataloaders, RDKit rendering and disk, and samples seen then vary per run. The eco implications recommend budgeting in tokens or images (eco F36); A only logs crops_seen.
- S0 is a week-1 single point of failure: a one-shot direct full fine-tune of Qwen3-VL-2B on 1.2M crops with no LR sweep. That contradicts A's own known negative ('direct full SFT of an OCR VLM diverged', ocsr F16). LR choices span 3 orders of magnitude (vlm F28), and the 13 h estimate uses the LoRA FLOP model.
- The Monday plan is circular and over-packed. The P-uspto label filter needs S0, which is not built until Tuesday night. 'Own CWU mining 2001-2024' and 800 human-checked USPTO 2025-26 DEV crops are multi-week data engineering, not day-1 tasks: the USPTO ODP layout is unverified (data F1 [UNVERIFIED]) and the download is likely multi-TB. Legal sign-off for MolRecBench-Wild eval use, which has no license, can block DEV v1.
- The proxy-validity gate uses Spearman ρ over only 8 recipes. The 95% CI for ρ=0.6 at n=8 is roughly −0.2 to 0.9, so the ρ≥0.6 gate cannot actually pass or fail anything.
- 4-7 of the 8 GPUs sit idle in weeks 1-4. The 360 s eval cap on 3,300 crops also makes A's own priors #13 and #14 effectively unevaluable: GTR traversal CoT is ~500 decode tokens per crop (≈1.6M tokens), and vote@8 multiplies that by 8.
- S2 setup in weeks 3-4 is several engineer-weeks compressed into ~1.5 weeks with no named staff: 150 in-house SAR-table gold, a harness R-group enumerator, 200k synthetic SAR tables in perturbed real layouts (no open generator exists, data F38), plus the S0-T build.
- Part of the scorer-gap list is stale against the current score.py: G4, G6, and the %inh part of G5.

**Ideas to graft**

- A warm start pinned once per stage (the tokenizer analogue), so trials optimize only the final real-data stage. Build it LoRA → selective FT as MolSeek did, with a 3-4-point LR mini-sweep on the otherwise idle GPUs.
- EVAL_WALLCLOCK_CAP with undecoded items counted wrong, so inference cost prices itself.
- Funnel: screen on 1 seed → replicate on 3 seeds with common-random-number seeds and stored incumbent per-doc files → CONFIRM as eval-only, one-bit feedback. Add a morning re-baseline of the incumbent on fresh seeds.
- Pre-loop calibration grid over budget lengths × 3 seeds, with explicit switching rules (σ_seed and rank flips).
- `run.sh --smoke` as a 60-s crash and throughput probe, with the runner as the sole writer of results.tsv.
- Run a single agent until a stagnation ping fires twice in a week; add agents only when forced.
- A hybrid-pivot trigger at a fixed week, tied to a named specialist baseline.

### Design B — score 5.5

**Strengths**

- The budget is in training tokens (7.5M), counted by the fixed trainer, not wall-clock. It is robust to multi-tenant contention, it stops throughput tricks, and verbose output formats pay for their own tokens (eco implications, F36).
- The trainer is frozen except for a small KNOBS block, and the data recipe is a separate file. GPU-side code rarely changes, so there are fewer OOM/NaN crashes and timeouts (eco F15 infra fragility), and data hypotheses are compared at fixed HPs (eco F42).
- CPU datagen runs pipelined with a content-addressed cache (~36 s for 17k crops on 16 cores) and an 8-min cap, so GPUs do not wait on rendering.
- The arithmetic is internally consistent. 17k crops at 20/s ≈ 14.5 min, ≈19 min per trial. 72 GPU-h gives ≈180 rung-1 trials + ≈22 rung-2 trials per night. Rung 3 (70k crops) ≈ 1 h, the 8B run (70k at 5/s) ≈ 3.9 h, and W0 (300k) ≈ 4.2 h.
- A cheap continuous companion metric, teacher-forced NLL (~0.5 min), plus an explicit 1-GPU fallback of ~38 trials per night.
- A weekly W_n re-cut with the frontier recipe advances the branch in data space at a known cost of 4.2 GPU-h.

**Fatal or major flaws**

- The generator library is the product, yet it is scheduled as 'generators v0' inside week 1, alongside containers, gates, receipts, score.py v1.1, W0 and noise runs. The library means RDKit/Indigo/CDK with ~40 style params, augment ops, a superatom-faithful label serializer plus the decode_contract expander, a Markush generator, label variants, and gate_data's InChIKey + pHash check on every materialized sample. A realistic first signal is week 2-3, not week 2.
- W0 is LoRA-only on 300k crops. The closest evidence, MolSeek LoRA-only on 96-192k samples, reached only 50.6 on USPTO (vlm F17). Each trial then measures the marginal value of 17k crops on a weak base. The thesis that 17k-crop recipe rankings predict scale leans on DataDecide (eco F41), which concerns pretraining corpora at 150M→1B. ρ is measured, but the core bet is untested.
- Stage B annotation is infeasible in the timeline. It needs ≥100 dev-fast, ≥300 confirm and 100 sealed double-annotated documents (≥8,000 records). At design C's estimate of ~2-3 curator-hours per document that is ~1,000-1,500 curator-hours by weeks 4-6, and none of it is budgeted. Week 4 also contains BindingDB↔USPTO alignment (no public alignment exists, data F19) and a real-layout SAR generator (none is open, data F38).
- The annotation throughput claim (2 chemists × 4 h/day ≈ 5k double-reviewed crops/week) is about 2× too high at ~30 s per molecule.
- A single greedy agent in nights 1-3 cannot keep 6 worker GPUs busy without explicit factorial waves. From night 4, three workers + critic + controller need ≈180 proposals per night, and the agent API spend is not budgeted.
- Hidden GPU cost: hard mining on 'mid-band incumbent NLL' (lever 15) needs an NLL pass over the pool after every incumbent change. That is ≈1M items × 430 tokens × 2N FLOPs ≈ 1.7e18 FLOPs, or ~2-3 GPU-h, and it is not in the budget.
- The rung-1 test applies p≤0.05 from a single-seed item bootstrap that ignores seed variance. It is anti-conservative, so many rung-2 confirmations will be spent on noise.
- Several of the listed score.py gaps are already fixed in the current file.

**Ideas to graft**

- A fixed TOKEN_BUDGET (or samples_seen) per rung, counted by the fixed trainer, instead of wall-clock.
- A frozen trainer with an editable KNOBS block (bounded ranges, at most 20% of trials) plus a separate data-recipe file. This means fewer crashes and data comparisons at fixed HPs.
- The recipe emits sample specs that a fixed renderer materializes on CPU, pipelined, content-hash-cached and time-capped.
- A harness-computed teacher-forced NLL as a ~30-s continuous tie-breaker and early signal, reported alongside the strict scalar.
- Data-gate rejection buckets (0 / <0.5% / ≥0.5%) instead of per-item feedback, so the gate is not a membership oracle, at no compute cost.
- Explicit 1-GPU fallback arithmetic.
- A weekly warm-start re-cut with the frontier recipe.

### Design C — score 3.5

**Strengths**

- R1 and R2 are budgeted in samples (20k and 60k crops) with time caps, not wall-clock. An R0 smoke test plus a one-pass pre-execution Code Fixer is the Amazon remedy for infra fragility (eco F15).
- Placebo arms in 1 of 10 R1 slots measure the gate's false-positive rate directly for ~10% of compute. This is the cheapest way to know whether the keep rule is honest.
- The ρ-audit design avoids range restriction (4 top / 8 middle / 4 bottom), reports ρ per lane, and makes LR, schedule and resolution changes R1-ineligible (eco F42).
- Per-rung arithmetic mostly checks out. R1: 20k at 25/s ≈ 13 min, ≈18 min per trial. R2: 60k at 8/s ≈ 125 min. R3: 500k at 4×8/s ≈ 4.3 h. RL-R2: 200 steps × ~42 s ≈ 2.4 h.
- The only design that states its staffing and estimates gold cost (200 record docs ≈ 400-600 curator-hours), with a pivot trigger at fewer than 120 docs by week 6.
- Resource caps: 200 GB scratch, 2 GB adapters, garbage collection after 72 h, and a kill at 1.5× the cap.

**Fatal or major flaws**

- The GPU plan is over-subscribed. All 8 GPUs are assigned (0-3 R1, 4-5 R2, 6 RL-R1, 7 eval; at night 4-7 run R3), yet several more jobs need GPUs. RL-R2 needs 4 GPUs × 2.6 h per day. The ρ-audit needs 16 R2 runs per week (~35 GPU-h). Every trainable-module change needs a 3-point LR sweep at R2 (3× cost). The monthly R2→R3 audit (16 R3 runs × ~20 GPU-h ≈ 320 GPU-h) would consume about half of the month's single nightly R3 slot.
- Hidden banking cost. R1 compares against 8 banked seeds and R2 against 3, so every ESTABLISHED change requires re-banking (≈8×0.3 + 3×2.2 ≈ 9 GPU-h). With a MAP-Elites archive of up to 8 parents, banking per parent multiplies this.
- R2 is the bottleneck. ~12 R2 slots per day, minus audits, second seeds, LR sweeps and re-banking, leave ~5 new candidates per day against ~85 R1 candidates per night. Most R1 compute feeds a queue that cannot drain, which makes for a poor cost per useful experiment.
- The RL lane gets 25% of R1 slots plus a dedicated GPU, but RL-R1 is only 150 GRPO steps (~4.8k prompts) of a 2B model in ~55 min. Published RL gains came from multi-hour to 70-h runs (vlm F22; vlm open question), so the expected signal per GPU-hour is low. The 20 s/step estimate omits the reference and old-policy log-prob passes and the vLLM LoRA slowdown (1.33-1.39×, vlm F31); ~30 s/step is more likely. The difficulty-filter pre-pass is unbudgeted, and the equal-compute SFT control doubles the cost. TRL's async GRPO does not take images (vlm F29), so the lane depends on a verl or EasyR1 integration.
- The engineering scope does not match the staffing. The orchestrator (scheduler, MAP-Elites, crossover, lane bandit, novelty embeddings, empirical-Bayes shrinkage, placebo bookkeeping, hierarchical cluster×seed bootstrap), 7 role prompts, bare-repo hooks, HMAC receipts, a logged datapool reader and RL infrastructure are to be built by one ML lead plus a cheminformatician. W3 turns on the full org and the RL lane in the same week. This is months of platform work and the most fragile design: Amazon lost 11 of 30 iterations to evaluator timeouts with only a 3-agent scaffold (eco F15).
- Agent API cost is unbudgeted: 4 specialists × ~3 proposals/h, plus a Code Fixer pass per diff, Critic audits and a Director, comes to hundreds of LLM calls per day.
- The first rung is the least powered of the three. dev-A has 1,500 items (SE_diff ≈ 0.8 pp), and with a 1.0-1.25 pp threshold it misses many real +1 pp effects that then never reach R2.
- Several of the listed score.py gaps are already fixed in the current file.

**Ideas to graft**

- Placebo arms (≈10% of screen slots) with target pass rates and automatic tightening of the gate.
- ρ-audit sampling spread across the Δ range, with per-lane ρ, plus marking LR, schedule and resolution edits as screen-ineligible.
- An R0 smoke test plus a one-pass pre-execution Code Fixer, once trials take 20 minutes or more.
- Empirical-Bayes shrinkage of screen deltas before agents see them.
- A mandatory equal-compute SFT-continuation control for any RL or MRT candidate.
- MI8 (sampled identity rate) as a cheap, representation-independent companion metric.
- An explicit staffing line and curator-hour estimate with a pivot trigger; scratch and adapter caps with garbage collection.

**Missing from all designs**

- No persistent eval server. None keeps a long-lived vLLM with hot-swapped LoRA adapters, so every trial pays ~4-5 min of container start, model and vLLM load, save and score. That is ≈23-28% of trial time in each design's own table. Restricting trial outputs to LoRA, merging only at promotion, would recover ~0.5 trials/h per GPU.
- No eval-repeatability noise floor. vLLM batched greedy decoding is not batch-invariant, so the same adapter scored twice can differ. None measures this before setting 0.5-1 pp keep thresholds.
- No matched non-agent baseline at equal GPU-hours, for example Hyperband or Optuna over the program.md knob table, as Claudini compared against Optuna (eco F36). The evidence says agents mostly tune hyperparameters (eco F28), so 'does the agent loop beat a scripted sweep per GPU-hour?' is the central compute-economics question, and none answers it.
- No total cost model. Agent API spend, engineer-weeks for the harness and generators, chemist-hours, and GPU cost are not added up. By my arithmetic, an 8×H100 node for 6 weeks is ≈8,064 GPU-h ≈ $22k at $2.69/h. Data-acquisition lead times are also missing: USPTO bulk CWU download and parsing (likely multi-TB; ODP layout unverified, data F1), and fetching PDFs for BioVista, ChemX and DocSAR, which ship IDs or DOIs rather than PDFs (data F23, ie ADDED).
- The proxy-validity study is underpowered. 8-16 rank pairs cannot tell ρ=0.6 from ρ≈0.2 (at n=16 the 95% CI for ρ=0.6 is roughly 0.1 to 0.85), and the high-fidelity points have their own seed noise. None budgets enough replicated high-fidelity runs for a decisive ρ, or says what to do when the CI straddles the threshold.
- No per-slot CPU, I/O or RAM reservation. Concurrent trials share CPU for image decode, augmentation and rendering, RDKit scoring and RL rewards, plus disk. This distorts wall-clock budgets and eval caps and adds unmodelled noise.
- No deployment-cost target (documents per day, $ per document at 8B), so the search can select vote@k, long traversal CoT or larger models that are unaffordable in production.
- No expected crash or timeout rate for the first nights, and no GPU-hours reserved for it. Only mitigations are listed, although Amazon lost 11 of 30 iterations (eco F15).

**Factual errors found**

- All three designs list score.py gaps that the current file (modified after the designs were written) no longer has. I checked this by running score.py with RDKit 2026.09.1. A's G4 (no slices, i.i.d. bootstrap), G6 (RDKit only printed) and 'no %inh test concentration' are fixed. B's #1-#3 (no slices, doc_id bootstrap, asymmetric unscorable exclusion), #5 (RDKit not asserted) and #6 are fixed: 'IC50 (µM)' now normalizes to 'ic50', 'mL·min−1·kg−1' to 'ml/min/kg', and %inh concentration and [lo, hi] ranges are supported. C's #2 (doc_id clustering), #5 (%inh concentration lost; 'mL/min/kg' ≠ 'ml·min-1·kg-1') and #9 (RDKit not enforced) are fixed. These gaps remain: C #4 (VALUE_REL_TOL 1e-3 rejects '63 nM' against pIC50 7.2 = 63.096 nM), '≤' collapsing to '<', pX conversion only when the unit is empty, string ranges ('10-30') parsing to None and silently leaving the gold denominator, µg/mL vs ng/mL unconverted, no target synonyms, and A's G1-G3 (dummy-label forms key differently; racemic keys equal to absolute; FragmentParent hides junk fragments).
- B: '2 chemists × 4 h/day ≈ 5k crops/week', with each crop double-reviewed, conflicts with data F5's ~30 s per molecule. 40 h/week is ≈4.8k single-pass or ≈2.4k double-reviewed crops.
- A lists Liger kernels as a throughput mitigation for Qwen3-VL, but vlm F31 says Liger's multimodal patch list did not include Qwen3-VL.
- A attributes '+5-14 pp on the abbreviation slice' to superatom-faithful targets via GTR. The ie F18 correction says the ~14-point figure is the full GTR-Mol-VLM against the second-best model, not an ablation of superatom labels alone. ocsr F29's retraining result does still support large gains.
- A's S0 estimate (~13 h on one H100 for full FT of 1.2M crops) applies the LoRA FLOP model (vlm F33: 6·N·T, with no weight gradients for the frozen base) and the LoRA planning rate to a full fine-tune. Full FT also computes weight gradients and optimizer updates, so I expect roughly 1.3× longer.
- B attaches 'vision-tower LoRA helps InternVL far more than Qwen (DATA F7)' to InternVL3.5-4B, but DATA F7 [UNVERIFIED] reports it for InternVL3-8B.
- B cites DataDecide's ~80% (eco F41) to argue that 17k-crop recipe rankings predict scale. eco F41 is about pretraining corpora ranked at 150M vs 1B, and the digest says no study measures this for small chemistry-VLM runs (eco open questions).
- C uses MolSight's ≈1.5 samples/s per A100 (8B LoRA) as a pure image-to-text throughput anchor without the vlm F33 checker caveat: that run also encoded SVG tokens and extra modules.

---

## Lens: rigor（评测严谨性）

**Ranking:** A, C, B

### Design A — score 7.5

**Strengths**

- The branch advances only after a pass on the hidden CONFIRM tier, which uses disjoint documents. A dev-only win is logged as DEV_KEEP and nothing is built on it. This answers the item-level winner's curse and the halved dev gain seen in the closest VLM analogue (eco F9, F16).
- Keep/discard is a three-stage funnel: a 1-seed screen, a 3-seed replicate with a 3/3 sign rule and a margin of max(0.6 pp, 2·σ_seed/√3), then CONFIRM. Seeds are owned by the runner and paired with the incumbent's. The expected best-of-28 null result (about 2 SE, metrics F24) is explicitly treated as below the bar for advancing.
- Every morning the incumbent is re-run on fresh seeds 3-5, and its stored files are replaced if it was lucky. This removes the regression-to-the-mean bias against challengers. It is the only design that handles incumbent luck.
- The S1 scalar is strict struct_f1 alone, and Markush stays a diagnostic until the R-label key is fixed. The inner-loop scalar therefore stays representation-independent while the agent changes output formats.
- It found real scorer gaps that reproduce and have since landed in score.py v1.2: R-label spelling (G1), racemic vs absolute stereo (G2), FragmentParent hiding junk fragments (G3, now a multifragment guardrail), multi-seed compare (G7) and an explicit --scalar flag (G10).
- Power is stated with the design effect (SE_diff about 0.6 pp, single-seed MDE about 1.7 pp; metrics F22). A day-4 calibration grid (budget × 3 seeds) comes with decision rules for the budget and thresholds fixed in advance.
- Deciding metrics are kept apart from diagnostics. The perturbation gap and canaries raise alarms but do not veto. A slice veto needs both an absolute drop and a drop of 2·SE. SEALED is run by humans only. A separate visible dev-viz set is used for error analysis, so DEV items are never exposed.
- Pivot triggers are explicit and measurable: ρ below 0.6 or 0.4, a CONFIRM pass rate below 30%, and the 8B model scoring below MolScribe's 62.3.

**Fatal or major flaws**

- The proxy-validity gate is underpowered. A Spearman ρ of 0.6 over 8 recipes has a 95% CI of roughly (-0.2, 0.9). Both rungs (2B for 15 min, 8B for 3 h) are single-seed, and no test-retest ceiling is measured, so the gate can neither confirm nor reject transfer.
- REPLICATE relies on a 3-seed-averaged paired bootstrap with p_not_better ≤ 0.05. score.py's pool_seeds plus document bootstrap ignores the between-seed (model-level) variance. In a null simulation (3,000 crops in 375 clusters, 3 seeds per side), the false-keep rate was 15% at a seed SD of 0.54 pp and 22% at 0.78 pp, against a nominal 5%. The 3/3 rule, the margin and CONFIRM mitigate this, but the claimed false-keep rate below 1% is a product that assumes independence and is never checked with placebo runs.
- The wall-clock TIME_BUDGET runs on a shared 8-GPU node alongside 3 parallel seed jobs, a promotion run and the data engine. Samples seen per run therefore depend on contention, which adds noise and confounds throughput with the recipe. Common-random-number pairing also breaks whenever a candidate changes the mixture or batch size.
- SEALED is used weekly and for the final choice among the top 3 states, so there is no untouched estimate of the shipped model. CONFIRM takes up to 8 one-bit queries a day for weeks and is refreshed only at stage gates, so it is adaptively overfit over time.
- The S1 metric only partly matches the downstream goal. About 40% of DEV is MolRecBench-Wild, drawn from general chemistry journals, and nothing reweights it by drug-likeness or domain. In-house crops arrive only in week 2. Gold crops leave out detection and identifier-linking errors.
- The effect-size priors in the search table are from-scratch effects from papers (+3 to +10 pp). The loop measures marginal 15-minute stages on top of S0, where real effects are likely below 1 pp to about 2 pp, close to the 1.7 pp single-seed MDE. The likely outcome is a loop whose decisions can be trusted but that keeps very few changes.
- S2 power is never computed (DEV is 400 tables). SAR-table errors are strongly clustered, since one misread header, unit or scaffold corrupts the whole table, so the effective sample size is close to the number of tables.
- The simplicity rule keeps a deletion with Δ ≥ -0.2 pp after CONFIRM. Each such keep carries about 0.6 pp of noise, so several of them can ratchet the frontier down.

**Ideas to graft**

- Advance the branch only on ESTABLISHED, meaning a pass on hidden CONFIRM. Keep dev-only wins as DEV_KEEP lineage entries for later recombination.
- Re-baseline the incumbent each morning on fresh seeds, and replace its stored per-doc files when it proves to have been lucky.
- Make strict struct_f1 the S1 scalar, and keep Markush diagnostic-only until a parsed-graph Markush key exists.
- Run a calibration grid before the first night, with rules fixed in advance for the time budget, σ_seed and the decision thresholds.
- Keep a visible dev-viz set excluded from every tier for error analysis. DEV shows only rounded aggregates and slice deltas, and CONFIRM returns one bit.
- Use runner-owned paired seeds (common random numbers) whenever the candidate leaves the data stream unchanged.
- Make vetoes require both an absolute drop and a drop of -2·SE, and keep the perturbation gap and canaries as alarms rather than vetoes.
- Keep a tried-and-failed descriptor table that rejects re-tests unless the hypothesis states what changed.

### Design B — score 5

**Strengths**

- The training budget is a fixed number of tokens counted by the harness-owned trainer. This removes throughput tricks (eco F5) and noise from node contention, and makes verbose output formats pay for their own tokens.
- It has the strongest leakage control of the three. gate_data checks the materialized samples, not only the code, against eval InChIKey block-1, pHash and document families. Hits are silently replaced, and the agent sees only a three-level rejection bucket, so it cannot use the gate as a membership oracle. A nightly audit checks the incumbent's full shard for zero overlap.
- The sealed set is run only at stage exits and in week 6, so it is the closest of the three to a true lockbox.
- Gold hygiene runs as a loop: weekly cleanlab-style disagreement mining, fixes by humans only, versioned gold and a re-baseline of the frontier (metrics F21).
- The Stage-B scalar is a macro average over patents, papers and SI, which limits the dominance of 500-compound patents. The Stage-A score is kept as a guardrail.
- Confirm queries are budgeted (3 per night, 60 per month) and 30% of items rotate in from a reserve set, Thresholdout-style. Feedback is rounded, and dev-fast gives no item-level output.
- KNOBS are restricted and data recipes are compared at fixed hyperparameters, which reduces confounding between data effects and hyperparameter effects.

**Fatal or major flaws**

- The branch advances on dev-only evidence at rung 2, while confirm runs only nightly on the top 3. With about 180 rung-1 trials a night across 6 workers, plus crossover in an elite pool, new trials are built on states that confirm later reverts. The design does not say how a revert works in a population.
- The scoring regime does not match deployment. Recipes are scored as a 17k-sample LoRA on W0 but shipped as a 300k-sample re-cut. Curriculum, augmentation-ladder and mixture levers act over a whole training run, so rankings from a marginal LoRA need not transfer. The rung-1 to rung-3 ρ check (4× tokens, still on W0) never tests the re-cut regime, and elites are not re-baselined after a re-cut.
- S_A gives Markush a weight of 0.20 on the CXSMILES key while TARGET_FORMAT is an agent lever. Until a parsed-graph Markush key exists, changes to format or label convention can move the scalar without any change in chemistry. Even after v1.2, a position-variation (m-section) Markush collapses to the core's InChIKey.
- The standard error is understated. The 0.41 pp figure assumes a pooled n of 3,000 and 5% discordance, but with slice weighting (Markush at 0.20 on about 300 items with high discordance) it is about 0.55-0.6 pp. The 0.8 pp PROMISING bar is then below 2 true SE, while the best of 180 null trials sits about 2.7 SE above zero.
- Many guardrails use fixed thresholds on small sets, all applied to every candidate: 100 canaries (>5 pp), a perturbation probe of about 300 items (>3 pp), a 300-item general probe (-2 points) and an NLL check (≤2σ). Each has roughly a 10-25% false-veto rate under the null, nothing controls the family-wise rate, and the candidates that survive are partly selected for luck on the guardrails.
- There is an adaptive channel from dev to training data. Per-MOSAIC-tag error rates on dev-fast (n ≥ 30) feed tag-targeted synthesis (lever 16), so the agent can fit dev-fast's tag composition and its noise.
- Candidates must win 3/3 pairwise against the incumbent's stored seeds, which were selected for being lucky, and there is no re-baseline, so the rule is biased against challengers. The pooled CI on seed-pooled counts leaves out between-seed variance and is anti-conservative.
- The NLL companion breaks under randomized-SMILES labels and label smoothing, both of which are in the lever list. An 'NLL no worse' veto can also reject real gains.
- Stage-B sizing borrows a design effect of 2.9, which rests on a hypothetical ICC of 0.1, for SAR-table records whose errors are far more correlated.

**Ideas to graft**

- Replace the wall-clock limit with a fixed training budget in tokens or samples, counted by the harness.
- Gate the materialized samples, not just the code, with silent replacement and bucketed rejection feedback, plus a nightly overlap audit.
- Treat the sealed set as a lockbox, run only at stage exits and once at the end.
- Run weekly disagreement mining, with human-fixed, versioned gold and a re-baseline after each fix.
- Macro-average the record scalar over document types (patent, paper, SI).
- Budget confirm queries and rotate items in from a reserve set.
- Add data-card columns to the lineage (pool shares, stereo, abbreviation and Markush fractions, mean tokens) so it is possible to diagnose why a recipe won.

### Design C — score 6.5

**Strengths**

- It has the most correct noise model. SE_tot = √(SE_diff² + σ_seed²(1+1/k)), estimated with a hierarchical bootstrap over clusters and seeds. It is the only design whose seed-pooled CI includes the between-seed variance.
- It states that re-seeding cannot remove the item-level winner's curse, because all seeds share the same dev items. Every R1 passer is therefore sent to fresh items (dev-B) before promotion.
- Placebo arms take 1 in 10 R1 slots and 1 R2 slot a week, with target pass rates and automatic tightening. The gate's false-positive rate becomes a measured number.
- R1 deltas are shrunk by empirical Bayes before they are ranked or shown to agents. This counters the max-of-85 artefact of about 2.45 SE (metrics F24).
- The ρ-audit is the best designed of the three. Its 16 candidates are spread over the Δ range to avoid range restriction, and it reports Spearman, Kendall and sign concordance with CIs, per lane. LR, schedule and resolution changes are not eligible at R1 because of short-horizon bias (eco F42), and R2 runs a fixed 3-point LR sweep.
- Every RL candidate must beat an SFT control with equal GPU hours. Rewards are built from harness primitives and computed on training labels only.
- It found the pX rounding-tolerance bug, which reproduces and is now fixed in v1.2. It also proposed an identifier-structure linking F1 and size caps against adversarial outputs.
- A claim enters the insights table as usable only after an agent other than its author reproduces it.

**Fatal or major flaws**

- There is no warm start. R1 trains Qwen3-VL-2B from the instruct base on 20k crops, a low-accuracy regime; MolSeek's LoRA stage reached only 50.6 on USPTO after 96k-192k samples (vlm F17). Intervention rankings there may differ from those at R3's 500k samples. R2 (60k crops at 8B) is also low-data, and the R2-to-R3 audit runs only monthly. The checkpoint RL-R1 starts from is never defined, and RL without SFT does nothing (ocsr F20), so the RL-vs-SFT verdict is ill-posed.
- S_crop gives Markush a weight of 0.20 on the CXSMILES string key. The plan defers the Markush scorer fixes to week 5, while the Representation lane changes Markush output formats (lever 10) from week 3. v1.2 has since fixed R-label spelling, but an m-section Markush still collapses to the core's InChIKey.
- The 'Hard' slice (stereo, subset C, low resolution) overlaps the other three slices. Items are counted twice, and the SE arithmetic ignores the resulting covariance.
- Guardrails have proliferated to about 12 vetoes, with fixed thresholds on small subsets. The OOD perturbation gap uses about 150 items with a 2 pp threshold, but its SE is about 3 pp, so it fires about 30% of the time under the null. The canary threshold (>3 pp) and the 200-item probe (-2 points) are similar. Nothing controls the family-wise rate, so power collapses and the survivors are selected partly for luck.
- sealed-T is used weekly to choose the release among the top 3 plus the incumbent, and also as an exit criterion (within 3 pp of confirm). It is adaptively reused and no longer an unbiased estimate.
- The S2 record tiers are far too small for keep/discard: dev-A has 40 documents, dev-B 60 and confirm-C 50. Record errors cluster at the table level, so the SE of record F1 is several pp and S2 decisions will be mostly noise.
- Requiring MI8 to agree in sign at R1 can veto real greedy gains, such as changes that sharpen the output distribution. MI8 is a sampled identity rate at T=0.7, a different estimand from greedy exact match.
- The organization is complex: 4 lanes, about 85 candidates a night, a MAP-Elites archive, crossover, and a bandit over lanes driven by noisy 'valid improvements per GPU-hour'. This multiplies forking paths. Placebos calibrate each gate's error rate but not the selection across lanes and the archive.

**Ideas to graft**

- Use SE_tot including σ_seed and a hierarchical cluster×seed bootstrap in gate.py, in place of the current seed-pooled compare.
- Add a second dev tier of fresh items (dev-B) for promotion decisions.
- Run placebo arms with target pass rates and automatic tightening, and report the gate's false-positive rate nightly.
- Shrink displayed and ranked deltas by empirical Bayes.
- Run a ρ-audit with candidates spread over the Δ range, Kendall and sign concordance, CIs and per-lane results. Make LR, schedule and resolution changes ineligible at the cheap rung, and run a fixed LR sweep at the higher rung.
- Give every RL or MRT candidate an SFT control with equal compute.
- Keep the previous stage's scalar as a veto slice when scope widens.
- Allow an insight to be used only after an agent other than its author reproduces it.

**Missing from all designs**

- The metric's validity for the downstream use is never checked. No design tests whether score gains turn into better analysis-ready data, for example by measuring agreement with BindingDB/ChEMBL on held-out documents, by training a QSAR model on extracted versus curated records, by timing curator corrections per document, or by measuring precision at fixed coverage. None sets an F-beta or cost weighting from the team's actual use, such as SAR modelling versus IP landscaping.
- Per-record confidence and abstention go unscored: no design measures calibration (ECE, precision at coverage), although curation triage and pseudo-labelling both depend on it.
- score.py v1.2 still has incentive bugs that no design flags; all three were reproduced with /tmp/claude-0/-home-user-autoresearch/453a67b8-16dd-54f4-9eb7-71994462b19d/scratchpad/rigor/probe2.py. (a) Qualitative records ('inactive', 'NT', '>10 µM' written as a string) are dropped from the gold recall denominator but still count in prediction precision. Reproducing them faithfully halves record precision, so the loop will learn to drop inactive data and bias SAR sets toward actives. (b) When one side was converted from pX, LOG_TOL_CONVERTED allows ±0.05 log10 (about ±12%), against ±0.1% for same-scale values. A misread 55 nM (gold 50 nM) re-expressed as pIC50 7.26 scores as correct, so emitting pX is an exploitable, representation-dependent advantage. (c) Position-variation Markush (CXSMILES m-sections, with the floating substituent as its own fragment) collapses through FragmentParent to the core's InChIKey: '*Cl.c1ccccc1 |m:...|' and '*OC.c1ccccc1 |m:...|' both key as benzene.
- Assay context is outside the record key. Swapping the biochemical and cellular IC50s for the same compound and target still scores record F1 = 1.0, and rat and human clearance share a key (both verified). Every design lists the missing assay/species field as a gap, but none says how assay type, species, cell line and mutant variant enter the S2 key or how the gold annotates them.
- Record-level clustering is never measured. All designs borrow the digest's hypothetical design effect (m=20, ICC=0.1). SAR-table errors are systematic (one header, unit or scaffold misread affects a whole table), so tiers should be sized and bootstrapped by table or document, with the ICC measured in calibration week.
- The cross-fidelity ρ-audit has no reliability ceiling. No design measures the seed-to-seed (test-retest) rank correlation within each rung, so a low cross-rung ρ cannot be told apart from attenuation by rung noise.
- Gold label quality is never measured. Inter-annotator agreement on record-level labels (what counts as a record, how targets are named, what identifies an assay) would set both the F1 ceiling and the label-noise floor against effects of about 1 pp. Convention mismatches, such as the abbreviation dictionary versus annotator expansions or undrawn versus racemic stereo, can produce 'gains' that only align conventions. No design requires the frozen abbreviation dictionary to reproduce the gold structure for every superatom in the gold.
- The public eval tiers do not match the team's domain. MolRecBench-Wild comes from 7 general chemistry journals and JPO/CLEF/UOB are scanned patents, yet nobody stratifies or reweights the scalar by drug-likeness or toward the team's own document mix.
- False vetoes are not controlled jointly. No design scales every guardrail threshold to its own SE, controls the family-wise false-veto rate across 6-12 guardrails, or measures how often each guardrail trips under the null (placebo runs could do this). The result is a hidden loss of power and selection on guardrail luck.
- Base-model contamination is never probed. No design compares zero-shot scores on documents from before and after Qwen3-VL's cutoff to check whether the base model saw MolRecBench-Wild source papers, USPTO benchmark images or BindingDB-aligned tables (an open question in the metrics topic).
- S1 scores gold crops, so detection and cropping errors never enter the inner-loop metric (BioVista AP-small is 0.05; ie F33 correction). A page-level joint metric, MolMole-style with IoU ≥ 0.5 AND exact match, appears at most as a nightly report.

**Factual errors found**

- B §2 and C §2 justify freezing detection and region cropping with '~0.99 mAP50' and 'close to saturated on digital PDFs' (DOC/DI F33). The digest's correction to F33 says detection is not saturated on BioVista, which is drug-discovery papers (MolDetV2 AP50 0.922, mAP 0.747, AP-small 0.050), or on scanned depictions (12.9 F1 points lost on ScannedRxn).
- B §4 presents 'Records cluster within documents, with a design effect of about 2.9 (MET F22)' as a property of its record tiers. Metrics F22 gives 2.9 only as a worked example (m=20 records per document, assumed ICC 0.1), not as a measured value.
- B §8 says 'Two chemists working 4 h/day produce about 5k crops/week' with every crop double-reviewed (DATA F5). The roughly 30 s per molecule in data F5 is for a single correction of a pre-annotation. With double review, the same hours give about 2.4k crops a week.
- C §4.2 cites ME F26 for 'MI8 has lower per-item variance than greedy 0/1'. ME F26 is about the signal-to-noise ratio of perplexity and bits-per-byte metrics across checkpoints, not about a sampled identity rate, which is also a different estimand from greedy exact match.
- Not a contradiction of the digest, but stale. The scorer-gap lists in all three designs (A G4/G5/G6/G9; B gaps 1-3 and 5-7; C gaps 2-5 and 8-10) describe score.py v1 (commit d545278). Versions v1.1 and v1.2 (commits 63ffacf and 155b44b, made after the designs were written) now include: a family-clustered NumPy bootstrap, slices, list-valued ranges, %inhibition test concentration, an enforced RDKit pin, a log tolerance for pX conversion, one key per R-label spelling, separate keys for enhanced (racemic/relative) stereo, multifragment, id-link and size-cap guardrails, --scalar, and seed pooling. The synthesis should plan against v1.2. Note that v1.2's seed pooling is anti-conservative: in /tmp/claude-0/-home-user-autoresearch/453a67b8-16dd-54f4-9eb7-71994462b19d/scratchpad/rigor/seedsim.py it gave 15-22% false keeps at a nominal 5% for seed SDs of 0.5-0.8 pp.

---

## Lens: domain（化学领域正确性）

**Ranking:** C, A, B

### Design A — score 6.5

**Strengths**

- Keeps the strictest stereo policy. The S1 scalar uses the stereo-aware full parent InChIKey, so a flat-drawn racemate predicted with invented stereo is wrong, which is what SAR needs. Stereo-agnostic block-1 F1 and Tanimoto are diagnostics only (metrics F5, F10).
- Found three real domain bugs in score.py that B and C missed, all reproduced by me with RDKit 2026.09.1. G1: [1*]c1ccccc1, [*:1]c1ccccc1 and *c1ccccc1 |$R1..$| give three different keys. G2: racemic |&1:| and |o1:| CXSMILES key the same as the absolute enantiomer, so rac and single-enantiomer records merge. G3: FragmentParent hides junk fragments (aspirin.CCO == aspirin).
- Handles salts correctly: parent is primary, and salt form is a separate diagnostic through the pinned ChEMBL Structure Pipeline (metrics F8), so the counterion information is not destroyed.
- Keeps Markush out of the decision scalar until the scorer can handle it. This avoids the broken CXSMILES key described under missing_from_all.
- S2 targets SAR tables directly: a table crop plus the scaffold crop plus the footnote strip, with a harness-owned, tested R-group enumerator behind an `rgroup_table_v1` output format. This matches ie F23: enumerate deterministically after the VLM reads the table.
- Superatom-faithful output, expanded through a frozen abbrev_v1.tsv inside the harness (ocsr F29, data F28). Fixed vetoed slices (chiral, abbreviation, >50 heavy atoms, scanned, in-house; each ≥300 items) stop hard chemistry being traded for easy images.
- Sound leakage practice for congeneric series: in-house Bemis-Murcko series stay in one tier, splits are by DOCDB family, and SEALED includes EPO/CNIPA/JPO patents and post-cutoff documents.
- Week-3 pivot to a hybrid, with MolScribe (MIT) or OCSRGlyph (Apache) as the structure tool, judged on SEALED against re-run baselines.

**Fatal or major flaws**

- Markush, the core of patent claims and about 49% of BioVista structures (data F23), gets no loop optimization in the 6-week window. It is delegated to MarkushGlyph, called 'permissively licensed', but MarkushGlyph's training index includes MolParser rows that are CC-BY-NC-SA (ocsr F26). That is a provenance risk for commercial use.
- The S2 record key is still thin: no assay type, species or cell line; no target-synonym or UniProt map and no mutant field; no µg/mL to molar conversion via MW; no ±SD or n; no binned activity. G5 lists some of these but makes no commitment. As a result, rat-vs-dog PK or HLM-vs-RLM swaps whose values still match go unpenalized.
- No compound-ID linking metric beyond the existing record_f1_by_label diagnostic. No name-to-structure (OPSIN) route for patent and SI examples defined by IUPAC name.
- Leaves VALUE_REL_TOL=1e-3 unchanged. That penalizes faithful rounding across representations; I verified that pIC50 7.2 becomes 63.096 nM and does not match a printed '63 nM'. No log-space or significant-figure rule is proposed.
- P-inhouse trains on 'in-house medicinal-chemistry papers, SI'. It does not address subscription-publisher TDM/AI-training rights (data F35 [unv]), which is a commercial-license gap.
- Several score.py gap claims are stale against v1.1 (commit 63ffacf, made after the design): G4 (slices and family bootstrap exist), G6 (the RDKit version is asserted) and parts of G5 (%inh test concentration exists; [lo,hi] list ranges are supported).

**Ideas to graft**

- Before keying, normalize dummy-atom R-labels ([n*], [*:n], $Rn$) to one canonical form (G1).
- Add an enhanced-stereo flag so that &/or groups (rac/rel) do not collapse onto absolute stereo (G2). Emit a per-compound stereo-status field.
- Add a predicted multi-fragment guardrail, and keep a full-structure (salt-form) match as a diagnostic next to the parent match, both via the ChEMBL Structure Pipeline (G3, metrics F8).
- Use a harness-owned deterministic R-group enumerator behind an `rgroup_table_v1` output format, with unit tests. The VLM reads the scaffold and the R-table and never generates enumerated SMILES.
- Keep the strict stereo-aware full InChIKey as the decision metric, with block-1 as a diagnostic only. Do not adopt an adaptive stereo rule.
- Feed S2 a table crop plus scaffold crop plus footnote strip, and add an explicit `--scalar struct_f1|composite` (G10).
- Keep each congeneric series (Bemis-Murcko scaffold) in one tier, and put EPO/CNIPA/JPO patents in SEALED.
- Build a multi-format S0 warm start (SMILES/CXSMILES/graph_json chosen by prompt tag), so that a change of representation is not handicapped.

### Design B — score 6

**Strengths**

- Best label-fidelity rules for real depictions: superatoms stay as drawn; `*` appears only for a drawn wildcard; USPTO `*` atoms are remapped to the visible R/X/Me; CDX wins over MOL when they disagree; stereo is labeled only where a wedge is drawn (data F2, F28; ocsr F29).
- Chemistry-aware data levers: decorative-wedge negatives (achiral molecules drawn with wedges, ocsr F30), wedge↔wavy/bold swaps, 2× chiral oversampling, scan and JPO degradation, large-molecule oversampling, and a Markush generator with R/m/Sg rates and template substituent text (data F13).
- Spots that generators can recreate eval molecules, for example by R-group enumeration on a pool scaffold. gate_data re-checks every materialized sample against eval InChIKeys and series scaffolds and silently replaces hits, so the gate cannot be used as a membership oracle.
- An unknown abbreviation is scored wrong and never collapsed to `*` (metrics F2).
- Stage B macro-averages over patents, papers and SI, so a single 500-compound patent cannot dominate. A record-precision floor of 0.85 at exit suits database curation.
- Synthetic SAR tables use perturbed real layouts, with nM/µM/pX, qualifiers, ranges, ±SD and footnotes. Scaffold plus R-table renders are labeled by RDKit enumeration. This follows Uni-Parser's finding that generic synthetic pages hurt (data F10).
- The only design that uses OPSIN name-to-structure, as a voter where an IUPAC name exists.
- Careful about licensing: an in-house detector because MolDetv2 is NC; frontier-API outputs used as labels only after legal reviews the output terms; CDK (LGPL) used unmodified in a separate process; ChEMBL CC BY-SA attribution noted.

**Fatal or major flaws**

- Puts Markush at 0.20 of the Stage-A decision scalar from day 1, scored by score.py's CXSMILES key. I verified that FragmentParent deletes position-variation (m:) substituent fragments: 'Cl anywhere on phenyl' gets benzene's InChIKey, the same key as 'Br anywhere' or plain benzene. Dropping substituents is therefore rewarded. B's v1.1 fix list does not identify this bug, and its IP5-M target (≥50 vs MarkushGlyph 58.2) compares two different scorers.
- Lever 18 labels counterion salts as the parent. This trains the model to discard drawn counterions, which contradicts its own 'labels describe what is drawn' rule and loses the salt form that solubility, dose and PK records need.
- Adopts an 'adaptive' stereo rule (stereo checked only when gold has stereo, ocsr F13). That lets invented stereo on flat-drawn racemates pass, which conflicts with its decorative-wedge lever and merges rac and single-enantiomer SAR.
- Measurement semantics are underspecified: no species, assay-type or cell-line fields, no UniProt or mutant map, no ID-linking metric, and ADMET/PK is effectively absent. Coreference such as 'Example 212' waits until Stage C (week 7+).
- Trains on BindingDB-aligned USPTO tables as 'weak real records'. BindingDB curates only a subset of each patent's measurements (targets and binding assays), so the incomplete labels teach the model to skip records.
- Frozen trainer, LoRA only. If stereo or representation is the bottleneck, as with coordinate-derived stereo (ocsr F28), data levers cannot fix it.
- Gap claims 1, 2, 3, 5 (assert), 6 (IC50 (µM) → ic50m, mL·min−1·kg−1, %inh concentration) and 7 (greedy matching) are stale against score.py v1.1.

**Ideas to graft**

- Adopt the label-fidelity rule set as harness-fixed policy: superatoms as drawn, `*` only for drawn wildcards, USPTO `*` remapped to visible labels, CDX preferred over MOL, stereo labeled only where a wedge is drawn.
- Add decorative-wedge negatives and wedge↔wavy/bold swap augmentation.
- Run gate_data over every generated sample (InChIKey block 1 plus series scaffold) and silently replace hits, closing the generator-recreates-eval-molecule leak.
- Score an unknown abbreviation as wrong, never as `*`.
- Macro-average record F1 over document types (patent, paper, SI) and require record precision ≥0.85 at stage exit.
- Build synthetic SAR tables on perturbed real layouts carrying pX/nM/µM, qualifiers, ranges, ±SD and footnotes. Label scaffold plus R-table renders by RDKit enumeration.
- Use OPSIN as a cross-check voter wherever an IUPAC name exists.
- Annotate about 500 in-house Markush items by week 4, because the public Markush sets are tiny.
- Train an in-house detector rather than use MolDetv2 (NC). Clear frontier-API ToS with legal before using API outputs as labels.

### Design C — score 7

**Strengths**

- Most complete measurement semantics. It proposes a frozen synonym→UniProt table plus a mutant/variant field (EGFR vs EGFR L858R/T790M vs ErbB1); assay type, species and cell line; %inh at a concentration; ranges; NT/inactive; ±SD; µg/mL needing MW; ADMET units; and records about '≤' collapsing into '<'. It also proposes log10 tolerance of 0.01, or reported significant figures, and I verified its example: pIC50 7.2 gives 63.096 nM, which does not match '63 nM'.
- Treats compound-ID linking as first-class: an id_struct_f1 (label plus key) metric, an S2 exit requiring ID-link F1 ≥0.85, and IdtVP identifier visual prompts (ie F13, correction-aware).
- Stereo ideas that go beyond oversampling: atom/bond JSON with coordinates so stereo is derived from geometry (ocsr F28, F30); RL shaping that penalizes stereo invented on achiral gold and penalizes a matching skeleton with the wrong stereo layer; a guardrail on the stereo gap.
- Plans deterministic parsed-graph Markush equality plus substituent-table F1 with no wall-clock MCS (metrics F13). Adds a graph-track diagnostic for the 2,632 MolRecBench-Wild items with no SMILES, and caps of 300 atoms per structure and 2,000 compounds per document.
- Best license hygiene: glyph-datasets minus its MolParser-derived rows; any NC sample found in a commercial run invalidates that run and all its descendants; ChEMBL share-alike and BindingDB CC BY noted; MolMole_Patent300 NC-ND kept eval-only; a per-sample license, track, source and family manifest.
- The S2 exit is measured against a strong hybrid baseline (PaddleOCR-VL table crops plus specialist OCSR plus zero-shot 8B linking, +15 pts). Curators get gold pre-filled from BindingDB/ChEMBL rather than using those databases as gold.
- Blocklists the Bemis-Murcko scaffolds of SAR documents from all training pools and from generator SMILES, and puts EPO/CNIPA documents in the sealed record tier.

**Fatal or major flaws**

- Same Markush hole as B: a 0.20 Markush slice in the S1 decision scalar from week 1. C's gap 6 assumes m/Sg sections are merely compared as strings, but FragmentParent actually deletes the m: substituent fragments (verified), so 'scaffold only' predictions score as correct on position-variation Markush. Its exit target (≥55 'CXSMILES-key EM' vs MarkushGrapher-2 at 53.2-53.7 and MarkushGlyph at 58.2) compares different scorers (ocsr F13, metrics F13).
- Proposes an 'adaptive' stereo rule for flat-drawn racemates and enhanced stereo in the primary metric. That would accept invented stereo in the decision scalar, which contradicts its own RL shaping. Strict keys plus an explicit enhanced-stereo flag are the domain-correct choice.
- Calls MarkushGrapher-2 a 'commercial-safe' recognizer and uses it as a label voter, but MG-2 was trained on 91k MolParser (CC-BY-NC-SA) samples (ocsr F25, ie F16). That is inconsistent with C's own exclusion of MolParser rows from glyph-datasets.
- No salt-form policy beyond FragmentParent, no OPSIN name-to-structure route for patent examples, and no binned-activity records.
- The org, ladder and RL machinery delays end-user value: the records loop starts in weeks 5-6 and the S2 exit is about week 8. ADMET/PK gold is never scheduled, although the scorer extensions mention ADMET units.
- Gap claims 2, part of 5 (%inh concentration; 'mL/min/kg ≠ ml·min-1·kg-1', which is now equal) and 9 (RDKit pin) are stale against score.py v1.1.

**Ideas to graft**

- Measurement key v2: a frozen target synonym→UniProt map plus a variant/mutant field; closed vocabularies for endpoint, assay type, species and cell line; µg/mL→molar via the record's structure MW; an ADMET/PK unit table; NT/inactive class; ±SD and n fields.
- Value match in log10 within 0.01 units, or at reported significant figures, plus an exact 'as-printed' diagnostic column.
- An id_struct_f1 (label plus structure key) metric and an ID-link F1 ≥0.85 exit criterion. Try IdtVP identifier visual prompts for linking.
- Offer coordinate-bearing atom/bond JSON as an output format with stereo derived from geometry. Use stereo-aware reward shaping that penalizes stereo invented on achiral or flat gold.
- Deterministic parsed-graph Markush equality plus substituent-table F1, with no MCS timeout. Add a graph-track diagnostic for gold that has no SMILES.
- Cap structure size and per-document compound count in the scorer.
- Remove MolParser-derived rows from glyph-datasets. On any NC contamination, invalidate the run and its descendants. Keep a per-sample license, track, source and family manifest.
- Blocklist the Bemis-Murcko scaffolds of SAR documents from training pools and generator SMILES.
- Gate S2 against a hybrid baseline of specialist OCSR plus table parser plus zero-shot VLM linking.

**Missing from all designs**

- Position-variation Markush is destroyed by score.py. STRIP_SALTS' FragmentParent keeps only the largest fragment, so the `*X` fragment that carries a CXSMILES m: section is deleted. 'c1ccccc1.*Cl |m:6:0.1.2.3.4.5|', its Br analogue and plain benzene all get benzene's InChIKey, and C1CC(*)CC1.*C |m:..| keys as *C1CCCC1. A scaffold-only prediction is therefore scored correct. Fix: skip the parent step for anything with dummy atoms or m/Sg sections, or key Markush by a parsed graph. Reproduced in /tmp/claude-0/-home-user-autoresearch/453a67b8-16dd-54f4-9eb7-71994462b19d/scratchpad/review_domain/probe2.py.
- Metal complexes and metallodrugs collapse. FragmentParent with metal disconnection keys cisplatin (N.N.Cl[Pt]Cl, or [NH3][Pt]([NH3])(Cl)Cl) as ammonia's InChIKey (QGZKDVFQNNGYKY), not as LXZZYRPGZAFOLE. Metal-containing structures need their own policy.
- Binned or categorical patent activity is missing: legends such as 'A: IC50 <10 nM; B: 10-100 nM; C: >100 nM' or '+++'. These are ubiquitous in pharma patents and need resolving the legend into an interval or qualifier record (ie implications). There is no record type, gold or slice for them.
- No production name-to-structure route. A and C ignore OPSIN entirely, and B uses it only as a data-engine voter. Patent and SI examples are mostly defined by IUPAC name, so 'transcribe the name, harness converts it with OPSIN' should be an allowed output format and an inference-time cross-check against OCSR with disagreement flags. None of the designs notes that US patents ship CWU MOL/CDX files that can be read directly at inference time, which bypasses OCSR for most USPTO structures.
- No stereo semantics beyond the key: no per-compound stereo status (absolute, relative, racemic, 'arbitrarily assigned', 'enantiomer 1 / peak 1' after chiral SFC); no E/Z policy for when gold lacks cis/trans (the strict InChIKey then penalizes a correct output); and no handling of atropisomers, which InChI ignores and which matter for kinase and KRAS chemistry.
- No ADMET/PK/physchem record schema or gold, even though the goal names them. That means species, dose, route, formulation and matrix (HLM/RLM/MLM, plasma), CYP isoform, pH for logD and solubility, kinetic vs thermodynamic solubility, and Caco-2 direction and efflux ratio. score.py's key has only a `conc` condition, so species or route swaps with matching values are unpenalized, and target is mandatory even for physchem.
- No replicate aggregation (n, ±SD/SEM, geometric mean, multiple values per compound per assay) and no cross-document dedup and merge (the same compound or measurement in paper, SI and patent). Both are needed for analysis-ready tables.
- No calibrated per-record confidence, abstention, or precision-at-coverage metric, and no provenance (page, bbox, table cell, footnote) from S1/S2. Chemists need triage and fast verification; MolScribe reports a 2× review speed-up from grounded output (ocsr F28).
- No inference-time value QC, such as CAPRICHO-style flags for 3.0 or 6.0 log-unit differences (nM/µM/mM confusion) and plausibility ranges per endpoint (ie F30).
- Upstream license provenance of 'permissive' weights is not checked. MarkushGrapher-2 trained on 91k MolParser CC-BY-NC-SA samples (ocsr F25), and MarkushGlyph's index includes MolParser rows (ocsr F26). Using them as baselines, voters or teachers for commercial labels launders NC data. A further unaddressed question is whether in-house copies of subscription journal papers and SI can be used for training, as opposed to inference (data F35).
- No gold-annotation scope rules: whether intermediates count; how reference or control compounds are handled; values cited from other papers ('data from ref 23'); tables duplicated between paper and SI; and 'Example N, Step 3' identifiers. Without these rules, precision is ill-defined.
- BindingDB/ChEMBL alignments are treated as if complete. They curate only a subset of each document's endpoints, and ChEMBL standard_value is normalized rather than as printed. Used as gold or training labels, they bias recall and the value-match rule.
- Stage-exit thresholds are anchored to published numbers (GPT-5.6-Sol 74.87, MarkushGlyph 58.2, MG-2 53.7) that were produced by other scorers (SMILES EM, MarkushGrapher cxsmi_equality). Baselines must be re-scored under score.py before any threshold is set (ocsr F13, metrics F3).
- Unit and endpoint vocabulary holes remain in v1.1, all verified. µmol/L and nmol/L are not recognized as molar. A pIC50 written with a unit ('M') is not converted. 'Kᵢ' normalizes to 'k', 't1/2', 'T½' and 'half-life' all differ, and '% inh. @ 1 µM' becomes 'inh1m'. ChemX SAR sets, used as S2 gold by A and C, are antibacterial MICs in µg/mL against organisms, so they need organism targets and MW-based conversion.

**Factual errors found**

- All three designs: their score.py gap lists were written against v1.0 (d545278) and several items are fixed in v1.1 (63ffacf, made after the designs). Now false: A-G4, A-G6 and parts of A-G5; B gaps 1, 2, 3, 5 (assert), 6 (IC50 (µM) → 'ic50m'; mL·min−1·kg−1; %inh concentration) and 7; C gap 2, part of 5 (%inh concentration; mL/min/kg mismatch, which I verified is now equal) and gap 9. v1.1 has family-clustered bootstrap, per-slice output, [lo,hi] ranges, a conc/conc_unit condition, unit stripping inside endpoint names, compound-unit canonicalization, optimal 1-D matching, symmetric exclusion of unscorable gold, and an RDKit pin enforced by sys.exit.
- C §8 calls MarkushGrapher-2 a 'commercial-safe' recognizer and uses it as a label voter. The digest says MG-2's phase-2 training used 91k MolParser samples (ocsr F25, ie F16), and MolParser data is CC-BY-NC-SA (ocsr F5, F38). Only the weights' license is Apache. B also lists MG-2 and MarkushGlyph as voters, and A calls MarkushGlyph a 'permissively licensed' baseline although its training index includes MolParser rows (ocsr F26).
- B §8: 'Two chemists working 4 h/day produce about 5k crops/week' with double review is inconsistent with data F5 (~30 s per molecule per correction pass). 40 person-hours gives about 4.8k single-pass or about 2.4k double-reviewed crops.
- A §4: DEV/CONFIRM draw 400 + 400 non-Markush, chiral-stratified crops from the BioVista-OCSR validation split (1:9 val:test). With 8,861 depictions the validation split is about 886 crops, and about 48.7% of BioVista structures are Markush/R-group (data F23). The validation split cannot supply about 800 non-Markush crops.
- C §4.2 cites ME F26 for MI8 (sampled identity rate) having lower variance. ME F26 is about likelihood/bpb metrics, not sampled accuracy. C §1 also groups ChemVLR with 'InChIKey identity' rewards, but ChemVLR's reward was Tanimoto=1.0 (ocsr F20), which is stereo-blind (metrics F5).
- B §4 gap 5 and C gap 9 recommend an 'adaptive' stereo rule citing ocsr F13. In the digest that is MolSight-2607's rule for benchmarks whose gold lacks stereo, not a recommendation, and metrics' open question flags that it hides real stereo errors. Using it as a primary metric for curated drug-discovery gold is unsupported.
