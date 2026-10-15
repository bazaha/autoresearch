"""
Null simulation: how often does a keep rule accept a change that does nothing?
Two modes, both on score.paired_compare:
  keep   (default) 3 vs 3 seeds, p_not_better <= 0.05: seed-pooled vs hierarchical (family x seed)
  screen 1 new seed vs 3 stored seeds, the screen rule delta >= +0.5 pp and p_not_better <= 0.20
Usage: python null_sim.py [trials] [screen | screen-latent]   (120 keep trials: about 10 minutes on one CPU core)
Result on 2026-10-15 (120 trials per row, nominal rate 5%):
  seed SD ~0.49 pp: pooled 2.5%,  hierarchical 0.8%
  seed SD ~0.56 pp: pooled 7.5%,  hierarchical 2.5%
  seed SD ~0.70 pp: pooled 10.8%, hierarchical 1.7%
Screen, 400 trials per row (nominal promising rate about 20%, Monte Carlo SE about 2 pp):
  `screen` (items flip independently between seeds), seed logit sd 0 / 0.03 / 0.05:
      v1.7.1 17.5 / 17.2 / 22.8%   v1.8 15.8 / 13.8 / 16.8%
  `screen-latent` (items stable across seeds, global seed effect dominates), sd 0.05 / 0.1 / 0.2:
      v1.7.1 35.2 / 36.5 / 38.0%   v1.8 22.5 / 21.7 / 17.7%
  (v1.8 borrows the missing global seed variance with a Student-t draw; harness/README.md section 6)
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import score

N_ITEMS, CLUSTER, SEEDS = 3000, 8, 3
SCREEN_MIN_DELTA, SCREEN_MAX_P = 0.005, 0.20

def make_run(rng, difficulty, seed_effect, item_noise=None):
    """item_noise None: each item flips independently between seeds (Bernoulli); a number: a
    latent model where an item mostly gives the same answer every seed, so the global seed
    effect dominates the run-to-run variation."""
    if item_noise is None:
        correct = rng.random(N_ITEMS) < 1 / (1 + np.exp(-(difficulty + seed_effect)))
    else:
        correct = difficulty + seed_effect + item_noise * rng.normal(0, 1, N_ITEMS) > 0
    return [{"doc_id": f"d{i}", "family_id": f"f{i // CLUSTER}", "slice": "all",
             "struct_tp": int(c), "n_pred_struct": 1, "n_pred_struct_raw": 1, "n_gold_struct": 1,
             "n_invalid_struct": 0, "n_multifrag_struct": 0,
             "record_tp": 0, "n_pred_rec": 0, "n_gold_rec": 0, "n_orphan_rec": 0} for i, c in enumerate(correct)]

def _runs(rng, seed_sd_logit, n_new, item_noise=None):
    score.BOOTSTRAP_SAMPLES = 300   # enough for a 5% decision; set here, not at import time
    fam = rng.normal(0, 0.8, N_ITEMS // CLUSTER).repeat(CLUSTER)
    difficulty = 1.4 + fam + rng.normal(0, 1.5, N_ITEMS)        # ~0.75 accuracy, clustered
    base = [make_run(rng, difficulty, rng.normal(0, seed_sd_logit), item_noise) for _ in range(SEEDS)]
    new = [make_run(rng, difficulty, rng.normal(0, seed_sd_logit), item_noise) for _ in range(n_new)]  # null
    return base, new

def trial(rng, seed_sd_logit):
    base, new = _runs(rng, seed_sd_logit, SEEDS)
    pooled = score.paired_compare(score.pool_seeds(base), score.pool_seeds(new))
    hier = score.paired_compare(base, new)
    return pooled["p_not_better"] <= 0.05, hier["p_not_better"] <= 0.05, hier["base_seed_scores"]

def screen_trial(rng, seed_sd_logit, item_noise=None):
    base, new = _runs(rng, seed_sd_logit, 1, item_noise)
    r = score.paired_compare(base, new[0])
    return r["delta"] >= SCREEN_MIN_DELTA and r["p_not_better"] <= SCREEN_MAX_P, r["base_seed_scores"]

def main(trials=120, mode="keep"):
    if mode == "screen-latent":   # latent model, item noise 0.3: the global seed effect dominates
        for sd in (0.05, 0.1, 0.2):
            rng = np.random.default_rng(int(sd * 1000) + 1)
            res = [screen_trial(rng, sd, 0.3) for _ in range(trials)]
            spread = np.mean([np.std(r[1]) for r in res])
            print(f"latent, seed logit sd={sd:.2f} (~{100*spread:.2f} pp seed SD): "
                  f"placebo promising rate={np.mean([r[0] for r in res]):.3f}")
        return
    for sd in (0.0, 0.03, 0.05):
        rng = np.random.default_rng(int(sd * 1000))
        if mode == "screen":
            res = [screen_trial(rng, sd) for _ in range(trials)]
            spread = np.mean([np.std(r[1]) for r in res])
            print(f"seed logit sd={sd:.2f} (~{100*spread:.2f} pp seed SD): "
                  f"placebo promising rate={np.mean([r[0] for r in res]):.3f}")
            continue
        res = [trial(rng, sd) for _ in range(trials)]
        spread = np.mean([np.std(r[2]) for r in res])
        print(f"seed logit sd={sd:.2f} (~{100*spread:.2f} pp seed SD): "
              f"false keeps pooled={np.mean([r[0] for r in res]):.3f}  hierarchical={np.mean([r[1] for r in res]):.3f}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 120, sys.argv[2] if len(sys.argv) > 2 else "keep")
