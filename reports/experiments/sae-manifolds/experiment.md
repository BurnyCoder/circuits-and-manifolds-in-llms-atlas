# Experiment: Can SAEs capture neural geometry? (synthetic benchmark)

Code [`src/atlas/entries/sae_manifolds.py`](../../../src/atlas/entries/sae_manifolds.py) (the docstring gives the full method) · site
data `site/public/data/sae-manifolds.json` · page `#/entry/sae-manifolds` · pre-stated criteria and dated amendments:
[`reports/planned-runs.md`](../../planned-runs.md).

## Question

When does a TopK SAE capture a concept manifold with a compact group of atoms, and when does it tile it (Bhalla et al. 2026, [arXiv 2604.28119](https://arxiv.org/abs/2604.28119))?

## Setup

48 manifold instances (8 types × 6 variants) embedded in d = 128; each training input sums L0 = 4 manifolds; TopK SAEs with 512 atoms for k ∈ {3…25}, 3,000 steps. Metrics follow the paper's App. E on superposed evaluation inputs: restricted R² (k_i greedy decoder directions reconstruct the manifold's contribution), support size and receptive-field spread, plus a regime label.

## Results vs published claims

<!-- table:replication -->
| Claim | Published | Ours | Criterion | Verdict |
|---|---|---|---|---|
| Subspace capture has a sparsity sweet spot | restricted R² (k_i atoms reconstruct each manifold from superposed codes) peaks near k = 4 (Bhalla et al. 2026, Fig. 4A) | mean restricted R² peaks at k = 6 (0.53; 0.42 at k = 3, 0.48 at k = 25); regime label 'capture' for up to 100% of 48 manifolds | restricted-R² peak at k ∈ {3, 4, 6} (amendment 2026-11-27: the paper's capture score; was 'capture fraction') | ✅ pass (post-hoc: metric switched to restricted R² after a full run whose log showed its peak) |
| Sparsity moves the SAE from shattering (small k) to dilution (large k) | low k: shattered across atoms; intermediate k: capture; high k: redundant atoms lose specificity (dilution) (Fig. 4B) | shattering 0% at k = 3 → 0% at k = 25; dilution 0% → 100% | dilution at the largest k ≥ 30 points above the smallest k and shattering higher at the smallest k (amendment: the pre-stated claim had the direction reversed relative to the paper) | ❌ fail (post-hoc: claim restated (direction corrected) after a full run) |
| Confirmatory: subspace capture has an interior sparsity optimum (fresh seeds) | restricted R² peaks near k = 4 (Bhalla et al. 2026, Fig. 4A) | k* = 6 on 3 selection seeds; on 3 held-out seeds R²(k*) − max(R²(3), R²(25)) = 0.075 [0.034, 0.121] (95% CI, seeds × instances); held-out R² by k: k=3: 0.42, k=4: 0.50, k=6: 0.54, k=8: 0.51, k=10: 0.51, k=14: 0.50, k=20: 0.48, k=25: 0.47. Originally pre-stated metric: capture fraction peaks at k = 3 (max 100%) | k* ∈ {4, 6} and CI of Δ > 0; partial if k* ∈ {8, …, 20} and CI > 0 (pre-stated 2026-11-28) | ✅ pass |
| Confirmatory: low k shatters, high k dilutes (fresh seeds) | low k: shattering; high k: dilution (Fig. 4B) | held-out seeds: dilution rises by 99 points [94, 100] from k = 3 to 25; shattering drops by 0 points | dilution rise ≥ 30 points with CI > 0 and shattering drop > 0; partial if both > 0 (pre-stated 2026-11-28) | ❌ fail |
<!-- /table:replication -->

## Interpretation

Both rows are post-hoc: the restricted-R² metric for claim 1 was adopted after a full run whose log already showed its
peak, and claim 2 was restated (direction corrected) after the same run. The curve is flat: the maximum
(`r2_best` at `k_best_r2` in the summary table) is nearly tied with k = 10 on the site's curve and only a few
hundredths above the ends (`r2_min_k`, `r2_max_k`), with one SAE seed per k and no CI. This does not support "degrades at both ends". The regime labels show the
paper's capture → dilution transition as k grows; we do not see shattering at the smallest k. Absolute restricted R²
is lower than the paper's. Details: [methods review](../../reviews/methods-review.md), C2.

**Confirmatory tests (pre-stated 2026-11-28).** Six fresh SAE seeds per k (training seed ctx.seed + 1000·(s+1) + k),
same 48 instances and steps.
- *Interior optimum: pass.* k* = 6 on seeds 0–2 (`k_star_sel`). On held-out seeds 3–5, R²(k*) exceeds the better of
  R²(3) and R²(25) by `peak_delta` with a two-level bootstrap CI (`peak_delta_lo`–`peak_delta_hi`) above zero. The
  held-out curve (`r2_held_by_k`) rises from k = 3 to k = 6 and declines slowly to k = 25, so the review's worry (a flat
  curve within single-seed noise) does not hold once seeds are averaged: the optimum is shallow but reproducible, and
  it lies at the paper's location (k ≈ 4–6).
- *Shattering → dilution: fail.* Dilution rises almost from none to all instances on held-out seeds (`dil_rise_held`,
  CI `dil_rise_lo`–`dil_rise_hi`), but shattering stays at zero at k = 3 (`shat_drop_held`). At small k our labels put
  nearly every instance in "capture" (few atoms per manifold, `capture_max`), not in many local atoms, even though
  restricted R² is lowest there. The paper's shattering regime may need smaller
  k, more atoms, or its exact phase-diagram thresholds (our regime label is a simplification).

## Limitations

- The regime label (support × spread thresholds) is ours; the paper shows a phase diagram.
- The original rows use one SAE seed per k; the confirmatory rows use six. No Ising-coupling recovery.

## All summary values

<!-- table:summary -->
| Key | Value |
|---|---|
| `k_best_r2` | 6 |
| `r2_best` | 0.532254 |
| `r2_min_k` | 0.420685 |
| `r2_max_k` | 0.482302 |
| `capture_max` | 1.0 |
| `k_best_capture` | 3 |
| `shatter_at_min_k` | 0.0 |
| `shatter_at_max_k` | 0.0 |
| `dilution_at_min_k` | 0.0 |
| `dilution_at_max_k` | 1.0 |
| `k_max` | 25 |
| `n_instances` | 48 |
| `steps` | 3000 |
| `k_star_sel` | 6 |
| `peak_delta` | 0.07466 |
| `peak_delta_lo` | 0.033693 |
| `peak_delta_hi` | 0.12072 |
| `dil_rise_held` | 0.986111 |
| `dil_rise_lo` | 0.9375 |
| `dil_rise_hi` | 1.0 |
| `shat_drop_held` | 0.0 |
| `n_seeds` | 6 |
| `r2_held_by_k` | k=3: 0.42, k=4: 0.50, k=6: 0.54, k=8: 0.51, k=10: 0.51, k=14: 0.50, k=20: 0.48, k=25: 0.47 |
<!-- /table:summary -->
