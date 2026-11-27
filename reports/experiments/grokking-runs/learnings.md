# Learnings: companion grokking runs

- **Download only what is needed.** The HF dataset holds ~360 MB of checkpoints, but the per-run analysis JSONs and
  `curves.npz` suffice for every statistic here, so the module downloads a few MB.
- **One run in the paper, 11 seeds here.** Nanda et al. analyse one main run. Cross-seed statistics turn "a handful of
  key frequencies" into a distribution (3–6 per seed).
