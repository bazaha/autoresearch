# Design A: a minimal, faithful port of autoresearch to a drug-literature extraction VLM

Citation key: (eco Fx) = autoresearch-ecosystem, (ocsr Fx) = ocsr-sota, (ie Fx) = doc-chem-ie, (vlm Fx) = vlm-finetune-practice, (data Fx) = data-sources, (metrics Fx) = metrics-eval-design. **[unv]** marks a finding the fact-checker could not verify. "ADDED" marks an item the fact-checker added.

**Assumptions.** Today is 2026-10-15. The team has one 8×H100-80GB node; the loop needs **one H100** in weeks 1-2, then up to 3 for parallel replicate seeds plus 1 for weekly promotions. The rest goes to one-off builds and the data engine. The agent is Claude Code in a sandbox. Use will likely be commercial, so the scored leaderboard trains only on commercial-safe data.

---

## 1. Thesis

Autoresearch works because of four mechanics: one editable file, a fixed time budget, one representation-independent scalar, and a keep/discard ratchet. Angle A ports exactly those four onto the narrowest valuable sub-task, **crop-level chemical structure recognition (OCSR) on drug-discovery depictions**. That is where end-to-end extraction breaks: BioMiner recalls activity values at 0.85 but structures at 0.58, its joint triplet F1 is 0.32, and removing its structure module drops it to 0.011 (ie F9, data F23). Gold is cheap (pre-annotated crops take ~30 s per molecule to check, data F5), public real-literature sets exist, the existing scorer already handles crops, and one experiment fits in about 20 minutes (§5). Two departures from autoresearch are forced by evidence (a third, the pinned warm start, is forced by the budget arithmetic in §5). The metric must leave the training process, because autoresearch protects `evaluate_bpb` with prose only (eco F4, metrics F29) and fine-tuning agents have trained on test sets and swapped in other checkpoints (eco F29). And "lower than best" must become noise-aware, because seed noise was ~22× the smallest kept delta in one session (eco F2) and the closest VLM analogue saw its dev gain halve on a locked confirmation set (eco F9). Populations, specialist multi-agent orgs, LLM critics and many-rung funnels wait until a measured failure demands them: in Agora the first three improvements gave 93% of the descent (eco F12), so a plain greedy loop running in week 1 beats a sophisticated one running in week 4. Scope widens only through stage gates (§2).

## 2. Staged scope and exit criteria

| Stage | What the inner loop optimizes | Scalar | Start |
|---|---|---|---|
| **S1: OCSR crops** | One-molecule crop to structure: stereo, abbreviations, salts. Markush is logged as a diagnostic only. | `struct_f1` (strict) | Week 1 |
| **S2: SAR tables** | Table crop plus its scaffold crop and footnote strip to compounds plus measurement records (IC50/Ki/Kd/EC50/%inh with target, relation, unit) | score.py composite: 0.5·struct_F1 + 0.5·record_F1 | Week 4 |
| **S3: pages and Markush** | Full paper, SI and patent pages to records with provenance; Markush CXSMILES plus R-tables | composite + Markush EM | After week 6 |

**Why tables come second, before pages or Markush.** 72.5% of BioVista activity values sit in tables (data F23). R-group enumeration is best done deterministically after the VLM has read the table (ie F23), so the harness owns the enumerator. Markush is covered by a permissively licensed baseline to buy rather than research: MarkushGlyph (Apache-2.0, Qwen3.5-2B LoRA, 58-61% on IP5-M; ocsr F26). Pages come last because page evaluation costs about 1 s per page even for sub-2B parsers (vlm F10), and their main extra difficulty, linking structures to values, is already exercised more cheaply in S2.

**S1 → S2 exit.** All of the following must hold:
1. At least 3 ESTABLISHED keeps, meaning keeps that passed the hidden CONFIRM tier.
2. The S1 recipe beats the S0 warm start on the SEALED tier, with the lower bound of the 95% CI above 0.
3. The proxy is valid: Spearman ρ ≥ 0.6 between "2B, 15 min" and "8B, 3 h" scores over at least 8 recipes.
4. Either the loop has plateaued (fewer than 1 ESTABLISHED keep in 2 nights), or the promoted 8B model reaches ≥ 75% on the MolRecBench-Wild dev slice (≈ GPT-5.6-Sol's 74.87; ocsr F15) and ≥ 0.75 on the BioVista-OCSR slice (MolParser-Mobile scores 0.80 but is non-commercial; ocsr F7).
5. The S2 gold set is ready and the score.py record gaps (G5 below) are patched.

S2 starts in week 4 even if item 4 is not met, provided items 1-3 hold.

**S2 → S3 exit:** ≥ 3 ESTABLISHED record-F1 keeps, a SEALED gain, and 60-120 page-level gold documents (ie implications).

## 3. File layout and the fixed/editable contract

**What is reused verbatim and what must change**

| autoresearch mechanic | Status |
|---|---|
| `autoresearch/<tag>` branch; commit per experiment; `git reset` on discard; baseline first | **Verbatim** |
| One agent-editable `train.py`, "everything is fair game" | **Verbatim**; boundary: no new data, checkpoints or network (eco F29) |
| `TIME_BUDGET` in prepare.py, excluding startup; schedules as a fraction of progress | **Verbatim**; 300 s → 900 s (§5) |
| NEVER STOP; kill at 2× budget; log-tail crash triage; no new deps; VRAM soft cap; simplicity criterion | **Verbatim**; thresholds in pp |
| `results.tsv` (untracked), `analysis.ipynb` frontier | **Kept**, more columns, written by the runner (eco F11) |
| Metric computed in train.py's process, grepped from stdout | **Changed**: separate eval container (eco F4, metrics F29) |
| "Keep if lower than best" | **Changed**: paired, 3-seed, hidden confirm (eco F2/F3/F8/F9) |
| One pinned val shard | **Changed**: DEV / CONFIRM / SEALED tiers (eco F16, metrics F24) |
| Train from scratch | **Changed**: allowlisted base + pinned warm start S0, built once like the tokenizer (ocsr F16, eco F42) |

**Layout**

```
research/drug-vlm/
  program.md          human-owned instructions (agent: read-only)
  prepare.py          constants + one-time builds: pools, eval tiers, S0 (read-only)
  train.py            THE ONLY FILE THE AGENT EDITS
  run.sh              agent's submit client: ./run.sh "hypothesis"  |  ./run.sh --smoke
  analysis.ipynb      frontier, per-slice and per-seed plots
  harness/            deployed read-only at /opt/dvlm-harness@<sha>
    score.py, test_score.py, golden_pairs.tsv   (~200 tricky molecule pairs)
    evaluate.py       harness-owned vLLM inference: base + adapter + infer_config -> pred.jsonl
    formats.py        closed set of output parsers -> molecules; abbrev_v1.tsv (frozen)
    keep_rule.py      screen / replicate / confirm logic and thresholds
    runner.py         gates, containers, receipts; sole writer of results.tsv + lineage.jsonl
  results.tsv, lineage.jsonl                    (runner-written; agent can read)
      results.tsv: commit stage seeds dev_score delta ci90_lo ci90_hi status invalid_rate
                   worst_slice_delta probe_delta crops_seen crops_per_s peak_vram_gb train_s eval_s description
/srv/dvlm/pools/   pre-rendered shards + manifest.sha256 + per-sample license tag (read-only to trainer)
/srv/dvlm/eval/    dev/ confirm/ sealed/ gold and images (readable only by UID evalsvc)
```

**prepare.py constants**

```python
BASE_MODEL  = "Qwen/Qwen3-VL-2B-Instruct@<pinned-rev>"; BASE_SHA256 = "..."
S0_CKPT     = "/srv/dvlm/s0/s0-2b-v1"      # multi-format warm start, built once per stage
TIME_BUDGET = 900                          # training seconds, excluding startup
EVAL_MAX_PIXELS = 1024*1024; EVAL_MAX_NEW_TOKENS = 1024; EVAL_WALLCLOCK_CAP = 360
OUTPUT_FORMATS = ("smiles", "cxsmiles", "molfile", "graph_json_v1", "rgroup_table_v1")
TRACK = "commercial"                       # which pools load_pool() exposes
```

**The train.py contract.** train.py reads data only through `prepare.load_pool(name)` and its seed only from `DVLM_SEED`. It must write `/out/adapter/` (LoRA, or full weights of the allowlisted base) and `/out/infer_config.json`: prompt ≤ 2,000 chars, `output_format` ∈ `OUTPUT_FORMATS`, `max_pixels` ≤ cap, decoding `greedy` or `vote` (k ≤ 8, T ≤ 1.0), `max_new_tokens` ≤ cap. Inside that, everything is fair game: LoRA, partial or full fine-tuning, losses (including MRT/GRPO with the harness's public reward functions on *training* pools), augmentation, on-the-fly rendering of pool SMILES, curricula, mixture weights, schedules. It may not edit other files, add data, use the network, download checkpoints, read eval paths, set seeds, or run code at evaluation time.

**Enforcement, mechanical rather than by instruction**

1. `prepare.py`, `program.md` and `harness/` are root-owned, 0444, `chattr +i`. The runner executes pristine copies from `/opt/dvlm-harness@<sha>`, never the agent's working tree.
2. **Diff gate.** If `git diff --name-only HEAD~1` is anything but `train.py`, the row is `gate_fail` and nothing runs (the tennis-xgboost `gate.sh` pattern, eco F10).
3. **Static scan.** ≤ 1,500 lines. Forbidden: `socket|requests|urllib|http|subprocess|ctypes` imports, path literals outside `/data` and `/out`, literal seeds, and any string ≥ 15 chars matching the runner's Bloom filter of eval SMILES, InChIKeys and document IDs.
4. **Training container.** `docker run --network none --gpus device=$G --read-only -v /srv/dvlm/pools:/data:ro -v $OUT:/out --tmpfs /tmp --pids-limit 512 dvlm-train@<sha>`, hard-killed at TIME_BUDGET + 600 s.
5. **Receipt** (the issue-#599 receipt made external, eco F4). The runner measures wall-clock itself, checks the adapter loads onto `BASE_SHA256`/S0, ‖ΔW‖ > 0, optimizer steps and samples seen > 0, and records SHA-256 of train.py, the pool manifest and the adapter.
6. **Scoring.** An eval container (UID `evalsvc`, `--network none`, the only place gold is mounted) runs `evaluate.py`, then `score.py score` and `score.py compare`, and emits signed JSON. The runner appends results.tsv. Scores never come from the agent's stdout.

## 4. The metric

**Primary scalar for S1.** `val_struct_f1` = score.py `struct_f1` on DEV: multiset match of stereo-aware standard InChIKeys of the RDKit FragmentParent+Uncharger parent. Invalid, missing or truncated outputs are wrong; extra structures are false positives. For one-structure crops this is exact match with a precision penalty. Higher is better; the agent sees it rounded to 0.1 pp.

**Representation independence, as with val_bpb.** `formats.py` turns every allowed format into an RDKit molecule (SMILES/CXSMILES parsed, molfile read, `graph_json_v1` atoms+bonds+wedges assembled; chain of thought allowed, only the last `<answer>` parsed; superatoms such as "Boc" expanded through the frozen `abbrev_v1.tsv`), and all go through the same InChIKey. Bits-per-byte cancels vocabulary size; the parent InChIKey cancels the output language, so the agent can move from SMILES to graph-traversal JSON on the same scale. Tanimoto never decides: it scores enantiomers as identical (metrics F5), and relaxed rewards can lower strict exact match (metrics F10).

**Vetoes** (block a keep even when the scalar rises):
- invalid-structure rate +0.5 pp (metrics implications); predicted multi-fragment rate +0.5 pp (G3);
- any slice (each ≥ 300 items: chiral, abbreviation, > 50 heavy atoms, scanned/low-res, in-house) with paired Δ below −1.5 pp *and* below −2·SE, so hard slices cannot be traded for easy ones (ocsr implications);
- a 300-item general doc-OCR/table probe down > 3 pp, guarding what S2 needs (vlm F25, F26).

Decoding stops at `EVAL_WALLCLOCK_CAP`; undecoded items score zero, so slow inference tricks cost score.

**Diagnostics, never deciding:** stereo-agnostic F1 (InChIKey block 1), Tanimoto, per-source F1, MolRecBench-Wild subset C (only ~150 items in DEV), IP5-M Markush EM, and the gap on 300 JPEG/rescaled DEV copies (a memorization alarm).

**Composition of the S1 evaluation tiers.** Splits are made by article or patent family, never by image.

| Tier | Source | Crops | License / note |
|---|---|---|---|
| DEV | MolRecBench-Wild 2026-08-19, SMILES-evaluable, subsets A/B/C (C capped at 150; only 298 exist, ie F37) | 1,200 | Card declares no license; built from CC-BY papers. Eval-only, after legal sign-off (data F26) |
| DEV | BioVista-OCSR validation split (1:9 val:test), non-Markush, chiral-stratified | 400 | MIT (ocsr ADDED: BioMiner) |
| DEV | USPTO 2025-26 grant CWU crops, MOL-derived and human-checked | 800 | US public domain; check 37 CFR 1.71 notices (data F37 [unv]) |
| DEV | In-house medicinal-chemistry papers, SI and patents (added in week 2 as DEV v2) | 600 | Proprietary |
| CONFIRM | The same four sources, disjoint documents | 800/400/400/400 | — |
| SEALED | Documents published on or after 2026-07-01; EPO, CNIPA and JPO patents; in-house project reports | 1,500 | Human-only |
| DIAG | Keylab OCSR-Benchmarks (corrected CLEF/JPO/UOB, ACS, USPTO-10K); IP5-M/M2S | — | Mixed / CC-BY; nightly, never decides |

DEV size is set for power. With n = 3,000 and discordance d ≈ 5-8%, SE_diff = √(d/n) ≈ 0.4-0.5 pp. Crops cluster at roughly 8 per document, which inflates this by about 1.3× (design effect; metrics F22). The result is SE_diff ≈ 0.6 pp, so the single-seed minimum detectable effect is about 1.7 pp.

**Gaps in score.py** (G1-G3 and G5's unit case reproduced with RDKit 2026.09.1 on the existing file):

- **G1.** `[1*]c1ccccc1`, `[*:1]c1ccccc1` and `*c1ccccc1 |$R1;;;;;;$|` get three different keys, which breaks representation independence for Markush and R-tables. Normalize to one dummy-label form before keying (needed before S2).
- **G2.** Racemic `C[C@H](N)C(=O)O |&1:1|` keys identically to the absolute enantiomer. Add an enhanced-stereo flag.
- **G3.** FragmentParent hides junk: `aspirin.CCO` matches aspirin. Add a predicted-fragment guardrail; score salt form as a diagnostic via the pinned ChEMBL Structure Pipeline (metrics F8).
- **G4.** No slices, and the bootstrap resamples documents i.i.d. although crops from one article are correlated. Add `cluster_id`/`slice` fields, a cluster bootstrap and per-slice paired deltas.
- **G5 (fix before S2).** Ranges ("10–30") parse to `None` and silently leave `n_gold_rec`; %inh has no test concentration; 5 µg/mL and 5,000 ng/mL get different keys; targets are only casefolded (no synonym table); no assay/cell-line/species field, NT/inactive class or ±SD parsing.
- **G6.** RDKit version printed, not asserted; no gold hash. Assert 2026.09.1 and emit hashes of gold, scorer and abbreviation table (metrics F7).
- **G7.** Add multi-seed `compare --base a,b,c --new x,y,z` (per-document counts averaged across seeds).
- **G8.** Unscorable gold is only counted: restrict DEV to scorable items and report coverage.
- **G9.** Pure-Python bootstrap: vectorize with NumPy so scoring stays under 30 s.
- **G10.** The composite silently becomes struct-only when a set has no records: add an explicit `--scalar struct_f1|composite`.

## 5. Budget per experiment

**Training throughput.** A 512² crop is 256 vision tokens in Qwen3-VL (32 px/token, vlm F32); with ~100 prompt and ~80 answer tokens, T ≈ 436. Cost per sample ≈ 6·N·T (6·2e9·436 ≈ 5.2e12) + frozen ViT (2·0.4e9·4·256 ≈ 0.8e12) ≈ 6e12 FLOPs. At 10-20% MFU of 989 TFLOP/s that is **16-33 crops/s**, inside the [unv] derived 10-45/s for 2B (vlm F33). Planning figure: **25 crops/s × 900 s ≈ 22k crops per run** (range 14k-30k), about a quarter epoch over a 90k real-crop pool, the size of MolParser's SFT set (ocsr F5) and sensible for a final-stage fine-tune. It is far too little to teach OCSR from a raw base: MolSeek's LoRA stage saw 96k-192k samples and reached only 50.6 on USPTO, and GTR trained on 1.3M samples for 3 epochs (vlm F16, F17). That is why every run starts from S0 and the loop optimizes the final real-data stage, where the evidence puts the big levers.

**Wall-clock per experiment**

| Phase | Seconds |
|---|---|
| Container start + load S0 | 60 |
| Training (fixed budget) | 900 |
| Save adapter | 20 |
| Eval: vLLM load (60) + decode 3,000 DEV + 300 probe crops, about 1.3M prefill and 0.3M decode tokens at about 9k tok/s for a 2B (olmOCR does 3,050 tok/s at 7B; vlm F11, F34 [unv]) (90) + ViT (30) | 180 |
| Score + 1,000-sample cluster-paired bootstrap | 60 |
| **Total** | **≈1,220 s ≈ 20 min** |

So **~3 runs/hour/GPU, 42 runs per 14-hour night (18:00-08:00)**. On one GPU, ~25% of candidates pass the screen and need 2 more seeds, i.e. 1.5 runs per candidate: **~28 candidates per night**, plus 2-4 CONFIRM evals (~2 min each) and a 1-hour morning re-baseline. With 3 GPUs running seeds in parallel: **42 candidates per night, each with 3 seeds**, same protocol.

**Other compute.** **S0** (once per stage, the analogue of tokenizer training): full fine-tune of Qwen3-VL-2B on 1.0M rendered + 0.2M USPTO crops, with SMILES / CXSMILES / graph_json targets chosen by a prompt tag as GOT-OCR2.0 does (ie F2), so a later format switch is not handicapped; ~13 h on one H100, ~1.7 h on eight. **S0-8B** (LoRA): ~5 h on 8 GPUs. **Weekly promotion**: current recipe on S0-8B for 3 h (32k-140k crops at 3-13/s [unv]).

**Fidelity levels**, deliberately few: 60-second smoke (OOM/NaN/crash, crops/s) → screen (1 seed) → replicate (3 seeds) → confirm (eval only) → weekly 8B promotion → human-run SEALED. No 5-minute rung: from a warm start, LR and augmentation effects need thousands of steps, and short budgets reward throughput tricks (eco F5, F42).

**Calibration (day 4, before the first night).** Baseline at TIME_BUDGET ∈ {300, 600, 900, 1800} × 3 seeds, plus 4 hand-made variants at 900 and 1800. Keep 900 unless σ_seed > 0.6 pp (go to 1,200) or the variant ranking at 900 differs from 1,800 (go to 1,800).

## 6. Keep/discard under noise

`keep_rule.py` (run by the runner) applies three stages. Pairing is per document, the bootstrap is clustered by document, and seeds come from the runner, so candidate and incumbent see the same data order (common random numbers).

1. **SCREEN** (seed 0 vs the incumbent's seed-0 per-document file): **PROMISING** if Δ ≥ +0.5 pp, `p_not_better` ≤ 0.25 and no veto. Otherwise discard and `git reset`.
2. **REPLICATE** (seeds 1, 2; the incumbent's 3 seed files are stored): **DEV_KEEP** if all three paired Δ > 0, mean Δ ≥ max(0.6 pp, 2·σ_seed/√3), and the 3-seed-averaged paired bootstrap gives `p_not_better` ≤ 0.05. This is the speedrun fork's 3/3 paired rule (eco F8) plus a margin.
3. **CONFIRM** (runner only; the agent sees PASS/FAIL): the 3 seed adapters on 2,000 hidden crops. PASS needs mean Δ > 0, `p_not_better` ≤ 0.10, and no CONFIRM slice down > 2 pp. PASS → **ESTABLISHED**: the branch advances and its 3 seed files become the incumbent. FAIL → `confirm_fail`, reset, and lineage keeps the idea as "dev-only" for later recombination.

**Simplicity rule.** A diff deleting ≥ 20 lines with mean Δ ≥ −0.2 pp is kept after CONFIRM; a diff adding > 60 lines must clear twice the thresholds.

**Winner's curse and dev overfitting.**
- With ~28 screens a night, the best null Δ is ~2.0 SE ≈ 1.2 pp (metrics F24: 1.64 SE at k = 12, 2.25 at k = 50), so a screen alone never advances the branch.
- **Morning re-baseline** on seeds 3-5: if the incumbent's mean is > 1 pp below its record, it was lucky; its stored files are replaced so it stops blocking honest candidates.
- **Ladder-style feedback** (metrics F25 [unv]): DEV rounded to 0.1 pp, CONFIRM one bit and ≤ 8 queries a day. At each stage gate the old CONFIRM folds into DEV and a fresh CONFIRM is drawn from the annotation queue.
- **SEALED** is human-only, weekly and at gates. Final selection scores the top 3 ESTABLISHED states on CONFIRM and SEALED and ships the best that passes; picking on validation alone cost 9-16 points on MLE-bench (eco F16).

Expected false-keep rate for a useless change: screen (~0.2) × 3/3 seed wins (~0.125) × margin × CONFIRM ≈ **< 1%**.

## 7. The agent's search space (program.md priors, prioritized)

| # | Edit to train.py | Expected effect | Evidence |
|---|---|---|---|
| 1 | Learning-rate sweep with separate LRs for LLM, merger and ViT; schedule as a fraction of the budget | +1-4 pp, cheap | LRs in published runs span about 3 orders of magnitude (vlm F28, F27) |
| 2 | Real-crop share of the mix: 10%, 30% or 50% | +3-10 pp on wild crops | ACS 0.15 → 0.37 → 0.46 [unv] (ocsr F17); real plus synthetic beats either alone (data F3) |
| 3 | Augmentation strength (scan noise, blur, JPEG, rotation, foreign text, wedge→wavy swaps) with a weak-to-strong curriculum | +3-8 pp | MolParser 40.1 → 69.5 → 76.9 (data F6); COMO augmentations (ocsr F30, F31) |
| 4 | Superatom-faithful targets: emit "Boc" as drawn and let the harness expand it | +5-14 pp on the abbreviation slice | GTR (ocsr F9, F29; data F28) |
| 5 | Oversample chiral molecules 2× (Stereo-200k, Apache-2.0) | +2-5 pp on the chiral slice | OCSRGlyph (ocsr F26, F30) |
| 6 | `max_pixels` 384² / 512² / 768²; padding that preserves aspect ratio | ±2-5 pp; cost scales with tokens | vlm F27, F32; ocsr F31 |
| 7 | LoRA on attention only vs MLP gate/up vs all linear layers; rank 16-128 | +1-3 pp, less forgetting | vlm F25 (+24.9/−0.6; +30.5/−2.1) |
| 8 | EMA, or averaging the last k checkpoints | +0.5-2 pp | EMA was about 63% of the gain in the speedrun fork (eco F8); souping (vlm F12) |
| 9 | Hard-example mining: oversample crops S0 gets wrong or where an ensemble disagrees (agreement 0.6-0.9) | +1-3 pp | MolParser (data F5); Logics-Parsing (ie F32) |
| 10 | General-data replay of 5-30% | Holds the probe veto | eco F9; vlm F26 |
| 11 | Batch size vs number of steps | +0.5-2 pp; may not transfer | eco F2 (biggest single win) vs eco F5 |
| 12 | COMO-style minimum-risk training (MRT) interleaved with MLE: N = 8-16, T = 0.5, R = 0.1·valid + 0.5·sim·valid + 0.4·exact | +3-10 pp on real crops | COMO +12-16 on JPO/ACS (ocsr F8; metrics ADDED) |
| 13 | Graph-traversal chain of thought (`graph_json_v1`) before the answer | +2-3 pp; slower decoding | GTR: +2.31 with CoT, +2.86-3.11 with traversal (ocsr F9) |
| 14 | Majority vote by InChIKey at test time, k ≤ 8 | +1-4 pp; eval time × k | MarkushGlyph MV@8 (ocsr F26) |
| 15 | SMILES-grammar constrained decoding | Lower invalid rate, +0.5-2 pp | vlm F14 |
| 16 | Tiling or a second view for molecules with more than 50 heavy atoms | +3-10 pp on that slice | ocsr F1, F31 |
| 17 | GRPO/DAPO with a binary InChIKey-identity reward, on prompts filtered by difficulty | +2-8 pp, or an exact-match drop | ChemVLR 92.7 vs 90.8/90.2 (ocsr F20); GSPO hurt exact match (ocsr F16, metrics F10) |
| 18 | Two-phase LoRA, then selective full fine-tuning | +1-3 pp, or instability | MolSeek (ocsr F16) |
| 19 | Unfreeze the top k ViT blocks at 10× lower LR | Depends on the base model | ocsr F17 [unv]; vlm F25 |
| 20 | Randomized vs canonical SMILES targets; CXSMILES output for depictions SMILES cannot express | ±1-2 pp; coverage | ocsr F27, F4 |
| 21 | Answer-only loss masking; label smoothing | 0 to +1 pp | Low prior |

**Order.** Weeks 1-2: items 1, 2, 3, 7, 8, 6, 4, 5, 9, 10. Week 3: 12, 13, 14, 15, 16. After that: 17-21.

**Known negatives in program.md:** direct full-parameter SFT of an OCR VLM diverged (ocsr F16); Thinking variants score lower on OCR (vlm F2); any RL edit must beat an equal-compute SFT control (ocsr implications).

## 8. Data strategy

**Fixed pools.** prepare.py builds these and hashes them. The agent sets the weights.

- **P-render.** 2M PubChem SMILES, filtered to exclude InChIKey block-1 matches with any eval tier. 1M are pre-rendered with RDKit (BSD-3), Indigo (Apache-2.0) and CDK (LGPL), using RanDepict-style random styles (data F31). Every image has two labels: superatom-faithful and expanded. The SMILES list is also exposed so train.py can render on the fly.
- **P-uspto.** MolScribe's 680K USPTO pairs (the HF repo is tagged MIT; data F3) plus our own CWU mining from 2001-2024. Labels are cleaned by remapping `*` to the visible R/X/Me, as COMO did. Pairs are dropped when S0 and two other recognizers all disagree with the MOL file (data F2). Any patent family that appears in the eval tiers or the classic benchmarks is excluded (data F30).
- **P-stereo.** Stereo-200k (Apache-2.0; ocsr F38).
- **P-inhouse.** Annotated in-house crops that are not used for evaluation. It grows weekly.
- **P-replay.** License-checked general document-OCR and table data.

**Two license tracks** (data implications). Every sample carries a license tag in its manifest.
- **Commercial track:** the pools above plus MarkushGrapher data (CC-BY-4.0; data F13) for S3.
- **Research track:** MolParser-7M SFT and MolGallery (CC-BY-NC-SA; data F4), and MolParser-Mobile as a teacher. It gets its own leaderboard and is never shipped.

**Label fidelity.** Superatoms kept as drawn (data F28); eval gold double-reviewed in Ketcher as in MolRecBench-Wild (ocsr F11); ~200 golden pairs must pass before any RDKit upgrade (metrics implications).

**Leakage controls** (metrics implications). Split by document and DOCDB patent family; dedup molecules by InChIKey block 1 across all pools and tiers; keep each in-house congeneric series (Bemis-Murcko scaffold) in one tier; perceptual-hash image dedup; SEALED only post-cutoff; eval molecules never rendered.

**Data engine** (offline, human-gated, outside the loop). Crops from commercial-track pages via the Docling figure classifier (MIT; ie F3); pre-annotation by an ensemble of the current best model, OCSRGlyph (Apache) and MolScribe (MIT); disagreements to chemists at ~30 s per molecule (data F5); weekly pool version bump, then re-baseline. Budget: 2 chemists × 4 days ≈ 1,600 double-reviewed in-house crops for DEV/CONFIRM/SEALED.

**Fixed:** pools, eval tiers, abbreviation table, base model, S0. **Agent-tunable:** sampling weights, augmentation code, on-the-fly renders, curricula, label view, losses.

## 9. Base model

- **Inner loop: Qwen3-VL-2B-Instruct.** Apache-2.0, OCRBench 858 (Thinking variant: 792), 32 px per token, supported by TRL, ms-swift, LLaMA-Factory, Unsloth and EasyR1 (vlm F1, F2, F29).
- **Promotion and production candidate: Qwen3-VL-8B-Instruct.** OCRBench 896. An 8B LoRA fits on one H100 (vlm F30).
- **Fallbacks**
  - InternVL3.5-2B or 4B (Apache-2.0; OCRBench 836/822; vlm F5), if it wins the week-1 bake-off.
  - Qwen3.5-2B, for Markush only. It is the base MarkushGlyph used (ocsr F26). Before using it, install flash-linear-attention and disable packing, because DeltaNet state leaks across packed rows (vlm F4).
- **Excluded**
  - Gemma 3: fixed 896 px / 256 tokens and the Gemma license (vlm F7).
  - MinerU2.5 and ChemDFM-X: AGPL (vlm F35).
  - Chemistry VLMs: ChemVLM 4.79 and ChemDFM-X 9.75 on MolRecBench-Wild v1 (vlm F15).
- **Bake-off on day 2:** zero-shot plus 2 h of identical SFT for Qwen3-VL-2B and InternVL3.5-2B, scored on DEV; ties go to Qwen.
- **Known risk.** VLM-only OCSR has landed at DECIMER level in one study (MolSeek; ocsr F16) and at 93% in another (GTR-VL; ocsr F10). The week-3 gate decides which applies here (§12).

## 10. Organization

- **One agent, one branch, one GPU**, e.g. `autoresearch/oct19-ocsr1`. `./run.sh "hypothesis"` blocks until the runner writes the row; `./run.sh --smoke` returns the log tail of a 60-second run for crash triage.
- **Lineage.** Runner-owned `lineage.jsonl` records hypothesis, diff stat, parent, seeds, every metric, status and failure reason. Each turn the agent sees the incumbent's row, its own last 15 rows, and a tried-and-failed table keyed by normalized knob descriptors (e.g. `lr=2e-4|lora=attn|r=64`); `run.sh` rejects a failed descriptor unless the hypothesis says what changed. Lineage was the strongest single ingredient in the CMU study, 16 vs 3 valid improvements (eco F11), and Amazon's agent re-tested known failures 6+ times despite prose (eco F15).
- **No LLM critic in weeks 1-4.** The runner pings the human after 12 candidates or 2 nights without an ESTABLISHED keep (a variant of Amazon's stagnation trigger, eco F15); the human writes a 2-5 sentence redirect into program.md. Agents change strategy in only 2.1% of transitions, and one redirect was worth up to 17 points (eco F30).
- **Human checkpoints.** Daily (15 min): results.tsv, frontier and slice plots, every ESTABLISHED diff. Weekly: SEALED eval, promotion run, program.md and data version bumps. Stage gates: the stage decision.
- **Scaling only if forced**, not before week 5, and only if the stagnation ping fires twice in a week: first a second independent agent on another branch with a different program.md focus, then a bounded elite pool of 5 (eco F13, F16).

## 11. Anti-reward-hacking and integrity

Beyond the enforcement in §3:

- **Seeds** come from the runner and are always paired; a keep must hold on 3 seeds, so seed lotteries cannot win (eco F3; eco ADDED Claudini seed search).
- **Visibility.** DEV: rounded scalar, slice deltas and guardrails, never per-item results. CONFIRM: one bit. A separate visible `dev-viz` set (200 crops, excluded from all tiers) is for error analysis.
- **infer_config** is schema-validated and its prompt scanned for SMILES-like strings.
- **Memorization checks.** Alarm when the perturbed-DEV gap grows > 3 pp; each week the human adds 100 canary crops of never-published molecules to DEV, whose accuracy must track real DEV (metrics implications).
- **Scorer strictness** (avoiding metrics F15, F16): invalid is wrong, no `*` collapsing, first prediction wins (no best-of-N), duplicates are false positives, missing documents score zero.
- **Weekly transcript sweep** by an LLM judge from another model family, looking for evaluator contact. It only flags, never scores, since judges can be gamed (metrics F35).

## 12. Top risks, mitigations and pivot triggers

| Risk | Mitigation | Pivot or abandon trigger |
|---|---|---|
| The 2B/15-min proxy ranks recipes differently from 8B/long runs (eco F5, F42) | Week-2 rank study; weekly promotion | ρ < 0.6: lengthen to 1,800 s. ρ < 0.4 at 1,800 s: drop the inner loop and run manual Hyperband sweeps |
| VLM-only OCSR hits a ceiling (ocsr F16) | Baselines: MolScribe (MIT), OCSRGlyph (Apache) | Week 3: promoted 8B below MolScribe's 62.3 on MolRecBench-Wild (ocsr F10) and a flat frontier → **pivot to a hybrid**: the specialist does OCSR as a tool, and the VLM loop moves to S2 linking and tables (ie F31) |
| DEV overfitting | Tiers, Ladder feedback, morning re-baseline, CONFIRM refresh | CONFIRM pass rate below 30% over 10 DEV_KEEPs → rotate DEV and raise thresholds |
| Too much noise | 3,000 crops, cluster bootstrap, 3 seeds | σ_seed above 1.5 pp → 3 seeds on every candidate (needs 3 GPUs), or grow DEV to 5,000 |
| Throughput below estimate (vlm F33 [unv]) | 60-second probe; Liger kernels and packing | Below 10 crops/s → TIME_BUDGET 1,200 |
| License contamination | Two tracks, per-sample tags, legal sign-off for eval-only MolRecBench-Wild use | — |
| Annotation bottleneck | Pre-annotation (~30 s per molecule) | — |
| Reward hacking | §3, §11 | Confirmed hack → freeze, patch the harness, re-baseline, mark the row `hack` |
| Strategy lock-in | Stagnation ping, weekly focus rotation | — |

## 13. Plan for weeks 1-6

**Week 1 (Oct 19-23): loop running by Friday night.**
- Mon: freeze the S1 spec; prepare.py builds P-render (~6 CPU-hours), P-uspto, P-stereo and DEV v1 (2,400 public crops); patch score.py G1-G4, G6, G7, G9; add `golden_pairs.tsv`; tests green.
- Tue: runner.py, containers, gate, run.sh; base-model bake-off; S0-2B builds overnight (~13 h).
- Wed: in-house annotation starts; 60-second throughput probe.
- Thu: calibration grid (§5) on 2 GPUs; set σ_seed and thresholds; program.md v1.
- Fri: first autonomous night on 1 GPU.
- *Deliverables:* harness + tests, S0, calibration report (crops/s, σ_seed, SE_diff, budget), ≥ 30 rows in results.tsv by Monday.

**Week 2 (Oct 26-30).** DEV v2 adds 600 in-house crops (re-baseline). Five nights, ~140 runs; daily reviews; program.md v2. Friday: first SEALED eval. Weekend: build S0-8B; rank study of 8 recipes (baseline, ESTABLISHED keeps, discards) at 3 h each on 8 GPUs. *Deliverables:* proxy-validity report, OCSR recipe v1, tried-and-failed table.

**Week 3 (Nov 2-6).** Week-3 ideas (MRT, traversal CoT, vote@k, tiling) with seed-parallel replicates on 3 GPUs. Promote the best recipe to 8B for 6 h → **OCSR-v1**; on SEALED, compare to MolScribe, OCSRGlyph, MarkushGlyph (IP5-M) and a human-run frontier VLM. In parallel: S2 gold (150 in-house SAR tables, ChemX Oxazolidinones/Benzimidazoles (MIT; ie F21), BioVista validation tables), G5 patched, harness enumerator for `rgroup_table_v1` with tests. *Friday: S1 gate and hybrid-pivot decision.*

**Week 4 (Nov 9-13).** Build S2 warm start S0-T from OCSR-v1 + 200k synthetic SAR tables (ChEMBL/BindingDB rows placed in perturbed *real* layouts, since generic synthetic layouts hurt; data F10) + USPTO XML example tables aligned to BindingDB patent rows (ie implications). Calibrate S2: ~1k vision + ~800 output tokens per table ≈ 5 tables/s ≈ 4.5k tables per 900 s run; DEV = 400 tables. First S2 night Wednesday. *Deliverable:* S2 baseline row vs the hybrid pipeline (OCSR-v1 + zero-shot VLM table reading).

**Week 5 (Nov 16-20).** S2 nights. If S1 and S2 are stable and the stagnation ping has fired, start a second single-agent loop on Markush crops (G1 fixed; IP5-M/M2S eval). *Deliverables:* S2 recipe v1; an evidence-based scale-up decision.

**Week 6 (Nov 23-27).** S2 gate, SEALED, 8B promotion → **Extractor-v1**. End-to-end report on 30 in-house documents with the hybrid page pipeline, page-level F1 as a report, not a loop objective (ocsr implications). Retrospective on which mechanics transferred. S3 design: 60-120 page-level gold documents; patent Markush claims vs exemplified compounds. *Deliverables:* Extractor-v1, retrospective, S3 plan.
