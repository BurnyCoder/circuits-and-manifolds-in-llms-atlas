# Experiment: cross-seed grokking statistics (companion runs)

Code [`src/atlas/entries/grokking_runs.py`](../../../src/atlas/entries/grokking_runs.py) · site data
`site/public/data/grokking-runs.json` (shown in the grokking entry) · fidelity 🔵 re-aggregated from the companion
project.

## Question

Do grokking and the Fourier circuit hold across many seeds, and do the controls (no weight decay, random labels) fail
as expected?

## Setup

Source: the companion project [visualizing-grokking-modular-addition-in-transformer](https://github.com/BurnyCoder/visualizing-grokking-modular-addition-in-transformer)
(MIT), whose 22 runs use Nanda et al.'s settings: 11 seeds with weight decay 1, 5 with weight decay 0, 3 random-label
controls, 3 with the paper's split and different inits. We download only small per-run files from the Hugging Face
dataset `BurnyCoder/grokking-modular-addition-runs` (`meta.json`, `curves.npz`, `analysis/{phases,final}.json`), not
the checkpoints, and compute: grokking delay (test-95% step ÷ train-99% step), key frequencies per seed, fraction of
neurons with > 85% variance explained by one frequency, and the controls' final test accuracy.

## Results vs published claims

<!-- table:replication -->
| Claim | Published | Ours | Pre-stated criterion | Verdict |
|---|---|---|---|---|
| Grokking replicates across seeds (companion project's 11 runs) | grokking with wd = 1 on a 30% split (Nanda et al. 2023, one run) | 11/11 seeds reach > 99% test accuracy; test-95% step ÷ train-99% step: median 57.2× (min 32.2×) | ≥ 9 of 11 seeds grok, with delay ≥ 3× in each | ✅ pass |
| A handful of key frequencies and single-frequency neurons in every seed | 5 key frequencies; 84.6% of neurons single-frequency | key frequencies per seed: 3–6; median 81.6% of neurons > 85% FVE | 3–7 key frequencies in every seed and median single-frequency fraction > 60% | ✅ pass |
| Controls: no weight decay / random labels don't generalize | no grokking without regularization; nothing to learn with random labels | wd = 0: best final test acc 21.5% over 5 runs; random labels: 0.9% over 3 | every wd = 0 run < 50% and every random-label run < 5% test accuracy | ✅ pass |
<!-- /table:replication -->

## Interpretation

Grokking is the rule, not a lucky seed: every weight-decay-1 seed groks with a delay of tens of × the memorisation time,
each with 3–6 key frequencies and mostly single-frequency neurons. No weight-decay-0 run generalises and random labels
stay at chance (1/113 ≈ 0.9%).

## Limitations

- These are the companion project's measurements re-aggregated, not independent training (our own 3 seeds are in
  `grokking`).

## All summary values

<!-- table:summary -->
| Key | Value |
|---|---|
| `n_runs` | 22 |
| `n_wd1` | 11 |
| `n_wd1_grokked` | 11 |
| `delay_median` | 57.214286 |
| `delay_min` | 32.222222 |
| `nkey_min` | 3 |
| `nkey_max` | 6 |
| `frac_neurons_median` | 0.816406 |
| `n_wd0` | 5 |
| `wd0_max_test` | 0.214677 |
| `n_rand` | 3 |
| `rand_max_test` | 0.008838 |
<!-- /table:summary -->
