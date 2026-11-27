# Experiment: grokking modular addition (trained here)

Code [`src/atlas/entries/grokking.py`](../../../src/atlas/entries/grokking.py) · site data
`site/public/data/grokking.json` · checkpoints `results/grokking/modadd_p113_seed{0,1,2}.pt` (gitignored; uploaded to
Hugging Face) · page `#/entry/grokking` · fidelity 🟢 recomputed (trained from scratch).

## Question

Does training a 1-layer transformer on (a + b) mod 113 reproduce grokking and the Fourier-multiplication algorithm of
Nanda et al. 2023 ([arXiv 2301.05217](https://arxiv.org/abs/2301.05217)), across seeds, and is weight decay required?

## Setup

| Setting | Value | Source |
|---|---|---|
| Model | 1 layer, d_model 128, 4 heads × 32, 512 ReLU MLP neurons, no LayerNorm, untied embed/unembed | paper §3 / official repo |
| Data | all 113² pairs, 30% train (seeded split), token "=" appended | paper |
| Optimiser | AdamW lr 1e-3, weight decay 1.0, betas (0.9, 0.98), full batch, 40,000 epochs | paper App. |
| Seeds | 0, 1, 2 (weight decay 1) + seed 0 with weight decay 0 (control) | — |

**Analyses.** Train/test curves; Fourier norms of W_E over training; key frequencies = frequencies carrying 90% of
‖W_E‖² in the Fourier basis; fraction of MLP neurons whose activation is explained by one frequency; causal ablation of
the key frequencies in the logits (project them out) vs removing the same number of random frequencies.

**Why these settings.** They are the paper's, so the result is a direct replication; 40k epochs leaves margin after the
paper's grokking at ~10k epochs.

## Results vs published claims

<!-- table:replication -->
| Claim | Published | Ours | Pre-stated criterion | Verdict |
|---|---|---|---|---|
| Generalization comes long after memorization (grokking) | train acc ~100% by ~1.4k epochs; test generalizes at ~10k (Fig. 2, wd=1, 30% split) | seed 0: train≥99% @ 200, test≥95% @ 7200; seed 1: train≥99% @ 200, test≥95% @ 6100; seed 2: train≥99% @ 200, test≥95% @ 13500 | test-95% epoch ≥ 3× train-99% epoch and final test acc > 99%, for every seed | ✅ pass |
| The embedding is sparse in the Fourier basis | W_E has ~6 non-negligible frequencies of 56 (§5.1) | ≤ 4 frequencies carry 90% of ‖W_E‖² (worst seed) | ≤ 10 frequencies carry 90% of the norm in every seed | ✅ pass |
| A handful of key frequencies; most neurons are single-frequency | 5 key freqs {14, 35, 41, 42, 52} in their seed; 84.6% of neurons > 85% FVE from one frequency | key freqs per seed: 1, 21, 49 \| 16, 17, 30, 55 \| 6, 43, 48, 56; 75.0% of neurons periodic (seed 0) | 3–7 key frequencies per seed and > 60% periodic neurons (exact frequencies are seed-dependent) | ✅ pass |
| The key frequencies are causally necessary | ablating key frequencies destroys performance; other frequencies don't matter (§5.2, Fig. 6) | test acc 1.000 → 0.012 (key removed) vs 1.000 (random removed) | key removal → < 20% acc; random removal → > 90% | ✅ pass |
| Weight decay is needed for grokking | no grokking without regularization in this setup (§5.3 / App.) | wd=0: final train 1.000, test 0.033 after 40000 epochs | wd=0 test accuracy stays < 50% | ✅ pass |
<!-- /table:replication -->

## Interpretation

All three seeds memorise within a few hundred epochs and generalise thousands of epochs later; the delay varies by seed.
Each seed uses its own small set of key frequencies (different sets across seeds, as in the paper), and removing them
destroys test accuracy while removing random frequencies does nothing. Without weight decay the model memorises and
never generalises within the budget. The companion entry `grokking-runs` confirms this across 11 more seeds.

## Limitations

- 3 seeds; the paper's restricted/excluded-loss progress measures are not recomputed here (they are in the companion
  project).

## All summary values

<!-- table:summary -->
| Key | Value |
|---|---|
| `epochs` | 40000 |
| `seeds` | 3 |
| `train99_epoch_s0` | 200 |
| `test95_epoch_s0` | 7200 |
| `final_test_acc_min` | 1.0 |
| `key_freqs_s0` | 1, 21, 49 |
| `key_freqs_all` | 1, 21, 49 \| 16, 17, 30, 55 \| 6, 43, 48, 56 |
| `frac_periodic_s0` | 0.75 |
| `n_freq_90_max` | 4 |
| `abl_clean` | 1.0 |
| `abl_key` | 0.012306 |
| `abl_rand` | 1.0 |
| `wd0_final_test_acc` | 0.033001 |
| `wd0_final_train_acc` | 1.0 |
<!-- /table:summary -->
