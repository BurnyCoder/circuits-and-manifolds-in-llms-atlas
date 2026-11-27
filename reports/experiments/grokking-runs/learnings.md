# Learnings: companion grokking runs

- **Download only what is needed.** The HF model repository holds a ~364 MB checkpoint file per run, but the per-run
  analysis files suffice for every statistic here (about 12 MB per weight-decay seed for the per-checkpoint metrics).
- **One run in the paper, 11 seeds here.** Nanda et al. analyse one main run. Cross-seed statistics turn "a handful of
  key frequencies" into a distribution (3–6 per seed).
- **Reuse the companion's preregistered definitions instead of inventing new ones.** For the progress measures we
  re-implemented its operationalizations (t1/t2 phase rules, 10% drop, 10× ratio) rather than writing our own criteria
  after seeing its results. Exact agreement on every seed's step counts is a strong code check. Because we had seen
  the outcome, the check is labelled as a cross-check, not a blind replication.
- **Axis range matters for phases.** On a 0–40,000-step linear axis the memorization phase (≈1,500 steps) is a sliver
  and its label collided with the next one. The progress chart shows steps up to 1.5 × t3 and numbers the bands, with a
  legend underneath.
