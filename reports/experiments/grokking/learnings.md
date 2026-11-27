# Learnings: grokking

- **Full-batch training is fast.** 40k epochs × 4 runs on 3,830 training pairs took ~26 min on the laptop GPU (see
  `runtime_s` in `results/grokking/metrics.json`).
- **Frequencies depend on the seed.** Each seed picks its own key frequencies (see `key_freqs_all`). The pre-stated
  criterion therefore tests the count (3–7), not the paper's specific set {14, 35, 41, 42, 52}.
- **Weight-decay-0 control.** It memorises (train 100%) and stays near chance on test, as in the paper. The companion
  project's 5 wd0 runs agree.
- **Reuse.** The companion project already has 22 runs with per-run analyses on Hugging Face, so we re-aggregated those
  (entry `grokking-runs`) instead of retraining many seeds.
