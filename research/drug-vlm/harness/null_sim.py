"""
Null simulation: how often does a keep rule accept a change that does nothing?
Compares seed-pooled vs hierarchical (family x seed) paired bootstrap in score.paired_compare.
Usage: python null_sim.py [trials]   (120 trials: about 10 minutes on one CPU core)
Result on 2026-10-15 (120 trials per row, nominal rate 5%):
  seed SD ~0.49 pp: pooled 2.5%,  hierarchical 0.8%
  seed SD ~0.56 pp: pooled 7.5%,  hierarchical 2.5%
  seed SD ~0.70 pp: pooled 10.8%, hierarchical 1.7%
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import score

N_ITEMS, CLUSTER, SEEDS = 3000, 8, 3

def make_run(rng, difficulty, seed_effect):
    p = 1 / (1 + np.exp(-(difficulty + seed_effect)))
    correct = rng.random(N_ITEMS) < p
    return [{"doc_id": f"d{i}", "family_id": f"f{i // CLUSTER}", "slice": "all",
             "struct_tp": int(c), "n_pred_struct": 1, "n_pred_struct_raw": 1, "n_gold_struct": 1,
             "n_invalid_struct": 0, "n_multifrag_struct": 0,
             "record_tp": 0, "n_pred_rec": 0, "n_gold_rec": 0, "n_orphan_rec": 0} for i, c in enumerate(correct)]

def trial(rng, seed_sd_logit):
    score.BOOTSTRAP_SAMPLES = 300   # enough for a 5% decision; set here, not at import time
    fam = rng.normal(0, 0.8, N_ITEMS // CLUSTER).repeat(CLUSTER)
    difficulty = 1.4 + fam + rng.normal(0, 1.5, N_ITEMS)        # ~0.75 accuracy, clustered
    base = [make_run(rng, difficulty, rng.normal(0, seed_sd_logit)) for _ in range(SEEDS)]
    new = [make_run(rng, difficulty, rng.normal(0, seed_sd_logit)) for _ in range(SEEDS)]  # same recipe: null
    pooled = score.paired_compare(score.pool_seeds(base), score.pool_seeds(new))
    hier = score.paired_compare(base, new)
    return pooled["p_not_better"] <= 0.05, hier["p_not_better"] <= 0.05, hier["base_seed_scores"]

def main(trials=120):
    for sd in (0.0, 0.03, 0.05):
        rng = np.random.default_rng(int(sd * 1000))
        res = [trial(rng, sd) for _ in range(trials)]
        spread = np.mean([np.std(r[2]) for r in res])
        print(f"seed logit sd={sd:.2f} (~{100*spread:.2f} pp seed SD): "
              f"false keeps pooled={np.mean([r[0] for r in res]):.3f}  hierarchical={np.mean([r[1] for r in res]):.3f}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 120)
