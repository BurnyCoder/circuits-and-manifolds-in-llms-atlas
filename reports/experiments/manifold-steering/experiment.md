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
| Steering along the manifold stays on the behaviour manifold | weekdays, Llama-3.1-8B: E_BC 0.34 (manifold) vs 0.93 (linear) | Qwen3-1.7B layer 22: E_BC 1.727 (manifold) vs 1.625 (linear) over 28 paths | manifold E_BC < 0.7 × linear E_BC | ❌ fail (post-hoc: layers 22–26 chosen after failing full-mode runs at layers 12–20) |
| Manifold paths pass through the intermediate days | manifold steering moves Mon → Tue → Wed; linear steering cuts across | fraction of path points predicting an intermediate day: 41% (manifold) vs 1% (linear) | manifold ≥ linear + 15 points | ✅ pass (post-hoc: layers 22–26 chosen after failing full-mode runs at layers 12–20) |
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

## Limitations

- The 6-D centroid span replaces the paper's 64-D PCA (too few prompts for it).
- The E_BC definition was kept as pre-stated, even though it may be dominated by how sharp steered distributions are.
- To do: a piecewise-linear control through the same centroids, a 64-D PCA with more prompts, and a layer fixed a priori.

## All summary values

<!-- table:summary -->
| Key | Value |
|---|---|
| `layer` | 22 |
| `accuracy` | 0.68254 |
| `E_manifold` | 1.72661 |
| `E_linear` | 1.62517 |
| `via_manifold` | 0.406593 |
| `via_linear` | 0.008242 |
| `n_paths` | 28 |
| `endpoint_by_layer` | L12 0.00, L16 0.00, L20 0.07, L22 0.86, L24 1.00, L26 1.00 |
<!-- /table:summary -->
