# Experiment: Steering along manifolds

Code [`src/atlas/entries/manifold_steering.py`](../../../src/atlas/entries/manifold_steering.py) (the docstring gives the full method) · site
data `site/public/data/manifold-steering.json` · page `#/entry/manifold-steering` · pre-stated criteria and dated amendments:
[`reports/planned-runs.md`](../../planned-runs.md).

## Question

Does steering along the curved weekday manifold move the output through intermediate days, and stay closer to normal behaviour, than straight-line steering (Wurgaft et al. 2026, [arXiv 2605.05115](https://arxiv.org/abs/2605.05115))?

## Setup

Qwen3-1.7B on day-arithmetic prompts it solves. Answer-day centroids from correctly answered prompts at layers 22/24/26, chosen by an endpoint check of where the subspace is causal. Periodic cubic spline through the centroids in their 6-D span; for every pair of days 2–3 apart, outputs along the spline arc vs the chord. Measured: the fraction of path points predicting an intermediate day, and the Bhattacharyya distance to the behaviour manifold.

## Results vs published claims

<!-- table:replication -->
| Claim | Published | Ours | Criterion | Verdict |
|---|---|---|---|---|
| Steering along the manifold stays on the behaviour manifold | weekdays, Llama-3.1-8B: E_BC 0.34 (manifold) vs 0.93 (linear) | Qwen3-1.7B layer 22: E_BC 1.648 [1.580, 1.717] (manifold) vs 1.504 [1.412, 1.599] (linear) over 112 paths from 28 source prompts (95% CIs over source prompts) | manifold E_BC < 0.7 × linear E_BC | ❌ fail (post-hoc: layers 22–26 chosen after failing full-mode runs at layers 12–20) |
| Manifold paths pass through the intermediate days | manifold steering moves Mon → Tue → Wed; linear steering cuts across | fraction of path points predicting an intermediate day: 46% [42, 50] (manifold) vs 2% [1, 4] (linear); piecewise-linear through the same centroids 44% | manifold ≥ linear + 15 points | ✅ pass (post-hoc: layers 22–26 chosen after failing full-mode runs at layers 12–20) |
| Confirmatory: the smooth manifold path beats a piecewise-linear path through the same centroids | the data-driven manifold geometry, not just visiting the right points, keeps behaviour on its manifold (Wurgaft et al.) | E_BC(spline) − E_BC(piecewise-linear), mean over paths [95% CI over source prompts]: L22 +0.062 [+0.057, +0.066]; L24 +0.089 [+0.083, +0.094]; L26 +0.098 [+0.092, +0.104] | CI below 0 at all of layers 22, 24, 26; partial at 1–2 (pre-stated 2026-11-28) | ❌ fail |
<!-- /table:replication -->

## Interpretation

Exploratory; both rows are post-hoc. The steering layers moved from 12/16/20 to 22/24/26 after two full runs at the
earlier layers had failed both claims; the endpoint check that motivated the move is now in the code and its result is
in the summary table (`endpoint_by_layer`: writing a day's centroid controls the answer only from layer 22 on). The
via-intermediate "pass" mostly reflects how the paths are built: the spline passes exactly through the intermediate
centroids, while a chord between two vertices of the centroid simplex never passes a third. The paper's quantitative
claim fails: both paths end far from the behaviour manifold, and the manifold path is slightly worse, unlike Llama's.
The verdict layer is the one with the best E_BC ratio, i.e. selected on the outcome. Details:
[methods review](../../reviews/methods-review.md), M2.

**Confirmatory test (pre-stated 2026-11-28): fail.** The review's control, a piecewise-linear path through the same day
centroids (reaching each intermediate centroid at the same t as the spline), evaluated at all three layers with 4
source prompts per day (112 paths per layer). The spline is further from the behaviour manifold than the piecewise-linear
path at every layer, with CIs over source prompts entirely above zero (`spline_vs_polyline`), and both are worse than
the straight chord. The polyline visits intermediate days about as often as the spline (row 2 of the table above).
So in Qwen3-1.7B, with a 6-D centroid subspace, the evidence is that passing through the day centroids moves the answer
through the intermediate days. Smooth interpolation between them adds nothing and slightly hurts. This is a negative
result for the paper's "data-driven manifold geometry" claim at this scale and setup, not a refutation of the Llama-8B
result (64-D PCA, many more prompts). Rows 1–2 now use the same 112 paths per layer (was 28) and carry CIs over source
prompts; their verdicts are unchanged.

## Limitations

- The 6-D centroid span replaces the paper's 64-D PCA (too few prompts for it).
- The E_BC definition was kept as pre-stated, even though it may be dominated by how sharp steered distributions are.
- To do: a 64-D PCA subspace with many more prompts, as in the paper.

## All summary values

<!-- table:summary -->
| Key | Value |
|---|---|
| `layer` | 22 |
| `accuracy` | 0.68254 |
| `E_manifold` | 1.647816 |
| `E_linear` | 1.50389 |
| `via_manifold` | 0.459478 |
| `via_linear` | 0.021291 |
| `n_paths` | 112 |
| `spline_vs_polyline` | L22 +0.062 [+0.057, +0.066]; L24 +0.089 [+0.083, +0.094]; L26 +0.098 [+0.092, +0.104] |
| `n_layers_spline_better` | 0 |
| `endpoint_by_layer` | L12 0.00, L16 0.00, L20 0.07, L22 0.86, L24 1.00, L26 1.00 |
<!-- /table:summary -->
