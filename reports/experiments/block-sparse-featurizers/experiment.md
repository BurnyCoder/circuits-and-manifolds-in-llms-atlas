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
| Confirmatory: blocks beat a TopK SAE on superposed (in-distribution) inputs | per-block R²: BSF 0.93–0.97 vs TopK SAE ≈ 0.53 (toy, Fel et al. 2026) | median per-concept R² on superposed inputs (L0 = 4), mean over 3 seeds: BSF 1.00 vs TopK SAE 0.45; BSF ahead in 3/3 seeds | BSF ≥ 0.8 and ≥ TopK + 0.2 with BSF ahead in all seeds; partial if BSF > TopK on average (pre-stated 2026-11-28) | ✅ pass |
| Confirmatory control: the orientation ring is learned (random-init InceptionV1) | the curve block reflects learned curve detectors | harmonics 1–3 of the top-16 PCs: trained 63% vs random-init network 73% (per-channel circular-shift null 63%, reported) | trained ≥ random-init + 10 points (pre-stated 2026-11-28) | ❌ fail |
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

**Confirmatory tests (pre-stated 2026-11-28).**
- *Toy, in distribution: pass.* Evaluated on superposed inputs (L0 = 4 concepts active, decoding only the concept's
  best block or its 4 most active atoms, without the shared decoder bias), the BSF keeps its advantage (`sup_bsf` vs
  `sup_sae`) in every one of 3 training seeds. The TopK SAE does worse in distribution than on isolated concepts, the
  same direction as the sae-manifolds finding that TopK fragments superposed manifolds.
- *Random-init InceptionV1: fail.* An untrained InceptionV1 (default PyTorch initialisation, weights checked non-constant)
  gives a higher harmonic share (`randinit_h123`) than the trained network (`curve_h123`). The per-channel circular-shift
  null reproduces the observed share (`shift_null_h123`), as predicted when the protocol was written: shifting a channel's
  tuning changes only its phase, so the share measures per-channel smoothness. Taken together, the InceptionV1 analysis
  as run here cannot detect the learned curve block; detecting it needs a trained BSF on natural-image activations, as
  in the paper.

## Limitations

- The curve 'block' is a PCA stand-in, not a trained BSF on InceptionV1.
- The toy concepts are exactly 3-D spheres matched to block size 4, which favours the BSF.
- The InceptionV1 stimuli vary only orientation and radius; with such stimuli a harmonic-share measure cannot separate
  learned from random features (see the confirmatory control).

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
| `sup_bsf` | 1.0 |
| `sup_sae` | 0.446478 |
| `sup_n_seeds` | 3 |
| `sup_bsf_wins` | 3 |
| `randinit_h123` | 0.731257 |
| `shift_null_h123` | 0.634947 |
<!-- /table:summary -->
