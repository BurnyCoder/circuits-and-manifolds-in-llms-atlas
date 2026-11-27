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
| Claim | Published | Ours | Pre-stated criterion | Verdict |
|---|---|---|---|---|
| Subspace capture has a sparsity sweet spot | restricted R² (k_i atoms reconstruct each manifold from superposed codes) peaks near k = 4 (Bhalla et al. 2026, Fig. 4A) | mean restricted R² peaks at k = 6 (0.53; 0.42 at k = 3, 0.48 at k = 25); regime label 'capture' for up to 100% of 48 manifolds | restricted-R² peak at k ∈ {3, 4, 6} (amendment 2026-11-27: the paper's capture score; was 'capture fraction') | ✅ pass |
| Sparsity moves the SAE from shattering (small k) to dilution (large k) | low k: shattered across atoms; intermediate k: capture; high k: redundant atoms lose specificity (dilution) (Fig. 4B) | shattering 0% at k = 3 → 0% at k = 25; dilution 0% → 100% | dilution at the largest k ≥ 30 points above the smallest k and shattering higher at the smallest k (amendment: the pre-stated claim had the direction reversed relative to the paper) | ❌ fail |
<!-- /table:replication -->

## Interpretation

The restricted-R² sweet spot replicates: capture peaks at intermediate k and degrades at both ends. The regime labels show the paper's capture → dilution transition as k grows, with most manifolds captured at small k and all diluted at large k. We do not see the paper's shattering at the smallest k, so the shattering-to-dilution claim fails on its shattering half. Absolute restricted R² is lower than the paper's.

## Limitations

- The regime label (support × spread thresholds) is ours; the paper shows a phase diagram.
- One SAE seed per k; no Ising-coupling recovery.

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
<!-- /table:summary -->
