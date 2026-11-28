# Experiment: Block-sparse featurizers

Code [`src/atlas/entries/block_sparse.py`](../../../src/atlas/entries/block_sparse.py) (the docstring gives the full method) · site
data `site/public/data/block-sparse-featurizers.json` · page `#/entry/block-sparse-featurizers` · pre-stated criteria and dated amendments:
[`reports/planned-runs.md`](../../planned-runs.md).

## Question

Do block-sparse featurizers keep multi-dimensional concepts together where TopK SAEs split them, and is InceptionV1's curve family a low-order Fourier block over orientation (Fel et al. 2026, [arXiv 2606.25234](https://arxiv.org/abs/2606.25234))?

## Setup

Toy: concept manifolds (spheres in their own 3-D subspaces of d = 128), L0 = 4 per sample; a BSF with blocks of 4 vs a TopK SAE with the same number of latents and active latents; per-concept R² from the best block / best atoms. InceptionV1: top-16 PCs of mixed3b responses to curve stimuli at 72 orientations; variance explained by orientation harmonics 1–3 vs a shuffled-orientation null.

## Results vs published claims

<!-- table:replication -->
| Claim | Published | Ours | Criterion | Verdict |
|---|---|---|---|---|
| Blocks capture multi-dimensional concepts that single-direction SAEs fragment | per-block R²: BSF 0.93–0.97 vs TopK SAE ≈ 0.53 (toy, Fel et al. 2026) | median per-concept R²: BSF 1.00 vs TopK SAE 0.64 (32 sphere concepts in 3-D subspaces of d = 128) | BSF median ≥ 0.8 and ≥ 0.2 above the SAE | ✅ pass |
| InceptionV1 curve responses form an orientation ring described by a few harmonics | one b = 16 block covers all orientations; harmonics 1–3 explain 59/18/12% (89%) | top-16 PCs of mixed3b curve responses: harmonics 1–3 explain 28/26/9% (63%; shuffled-orientation null 95th pct 3%) | harmonics 1–3 ≥ 50% and above the null's 95th percentile | ✅ pass |
| Control: random mixed3b channels are less harmonic than the curve neurons | the orientation ring is specific to the curve family | orientation harmonics 1–3 explain 71% of the 10 curve neurons' responses vs 65% for 10 random non-curve channels | curve neurons ≥ random channels + 10 points | ❌ fail (post-hoc: control added after the methods review) |
<!-- /table:replication -->

## Interpretation

The toy claim passes: the BSF reconstructs each concept from one block much better than the TopK SAE does from the
same number of atoms, although the toy shows each concept alone, the out-of-distribution evaluation the sae-manifolds
amendment rejected, and its 3-D spheres match the block size. The InceptionV1 claim passes its pre-stated criterion,
but that criterion cannot fail: the stimuli vary only in orientation and radius, so any smooth network's responses are
periodic in orientation, and the shuffled-orientation null destroys that smoothness. The control added after the
methods review confirms this: orientation harmonics explain almost as much of 10 random non-curve channels' responses
as of the 10 curve neurons (`random_channels_harm_share` vs `curve_neurons_harm_share` in the summary table), short of
the 10-point margin, so the control row fails. Details: [methods review](../../reviews/methods-review.md), M13.

## Limitations

- The curve 'block' is a PCA stand-in, not a trained BSF on InceptionV1.
- The toy concepts are exactly 3-D spheres matched to block size 4, which favours the BSF.
- To do: random-init InceptionV1 control, a smoothness-preserving null, and toy evaluation on superposed inputs.

## All summary values

<!-- table:summary -->
| Key | Value |
|---|---|
| `bsf_r2_median` | 1.0 |
| `sae_r2_median` | 0.644403 |
| `n_concepts` | 32 |
| `curve_h1` | 0.281271 |
| `curve_h123` | 0.628774 |
| `curve_null_p95` | 0.02613 |
| `curve_neurons_harm_share` | 0.705829 |
| `random_channels_harm_share` | 0.648308 |
<!-- /table:summary -->
