# Experiment: The Jacobian lens / global workspace (open model)

Code [`src/atlas/entries/jlens.py`](../../../src/atlas/entries/jlens.py) (the docstring gives the full method) · site
data `site/public/data/jlens.json` · page `#/entry/jlens` · pre-stated criteria and dated amendments:
[`reports/planned-runs.md`](../../planned-runs.md).

## Question

Does the Jacobian lens surface unspoken intermediate steps better than the logit lens, and does swapping J-space coordinates swap concepts, in an open 1.7B model (Gurnee et al. 2026, [Transformer Circuits](https://transformer-circuits.pub/2026/workspace/index.html))?

## Setup

Qwen3-1.7B with Neuronpedia's pre-fitted J-lens. Slice grids for J-lens vs logit lens; best rank of the intermediate token (Texas, France) at the last position. Concept swaps at the country token in middle layers, both the pre-stated coordinate swap and a paper-style sparse NNLS coefficient swap (amendment), each vs a random control.

## Results vs published claims

<!-- table:replication -->
| Claim | Published | Ours | Criterion | Verdict |
|---|---|---|---|---|
| The J-lens surfaces unspoken intermediate steps better than the logit lens | J-lens beats logit / tuned lens; intermediates (21, 42) become readable in middle layers (Claude) | best rank of the intermediate token at the last position: median 4 (J-lens) vs 9 (logit lens); top-10 for 3/3 vs 2/3 | J-lens median rank lower than the logit lens's and ≥ half of intermediates reach the top 10 | ✅ pass (post-hoc: arithmetic case removed from the metric after its ranks had been seen) |
| Swapping J-lens coordinates swaps the concept | France → China swaps change capital/language/continent answers together (76/192 top-1 across functions, Claude) | layer 11: target answer in top-5 for 12% [2, 47] of 8 prompts (random direction 0% [0, 32]; Wilson 95% CIs) | ≥ 50% and ≥ 30 points above the random-direction swap | ❌ fail |
| Swapping J-space coefficients (paper-style sparse NNLS decomposition) swaps the concept | category ("think of a {category}") swap with the J-space component: 59% top-5 vs 5% for the non-J component (Claude) | layer 9: target answer in top-5 for 12% [2, 47] of 8 prompts (random vector 0% [0, 32]; Wilson 95% CIs) | ≥ 50% and ≥ 30 points above the random-vector swap (amendment 2026-11-27, same thresholds) | ❌ fail (post-hoc: paper-style swap added after the pre-stated swap failed) |
| Confirmatory: on 24 fresh two-hop prompts the J-lens ranks the unspoken intermediate better than the logit lens | J-lens beats the logit lens at surfacing unspoken intermediates (Claude) | J-lens better on 16, logit lens better on 6, tied on 2 of 24 prompts (sign test p = 0.0525); median best rank 5 vs 8; top-10 for 15 vs 14 | J-lens better on more prompts than the reverse and two-sided sign test p < 0.05 (pre-stated 2026-11-28 before running) | 🟡 partial |
<!-- /table:replication -->

## Interpretation

Suggestive, not established. The original lens row is post-hoc: the arithmetic prompt was removed from the metric
after its ranks had been seen, and those ranks went against the J-lens (first-digit tokens of "21" and "42" at ranks
1360 and 1088 under the J-lens vs 16 and 42 under the logit lens; the review recomputed the row with them included and
it would still pass). To get past n = 3, a confirmatory test on 24 fresh two-hop prompts was pre-stated (commit
`8d1376e`) and run unchanged: the J-lens gives the intermediate a better best rank on most prompts, but the sign test
lands just above the pre-stated p < 0.05 (`confirm_sign_p` in the summary table), so the verdict is partial. The
top-10 counts are nearly equal, so the advantage is in moving intermediates a few ranks up rather than surfacing ones
the logit lens misses. Both swap variants fail (1/8 at best vs 0/8 random), the paper-style one being post-hoc as well.
This matches the open replication avalur/JLens, where selectivity did not reproduce up to 9B.
Details: [methods review](../../reviews/methods-review.md), M11.

## Limitations

- Eight swap prompts are too few; the confirmatory rank test has 24 prompts and one layer range.
- Qwen3's digit-level tokenization makes multi-digit intermediates unrankable.

## All summary values

<!-- table:summary -->
| Key | Value |
|---|---|
| `n_layers_lens` | 27 |
| `inter_median_rank_j` | 4.0 |
| `inter_median_rank_logit` | 9.0 |
| `inter_top10_j` | 3 |
| `inter_top10_logit` | 2 |
| `n_inter` | 3 |
| `swap_layer` | 11 |
| `swap_top5` | 0.125 |
| `swap_random` | 0.0 |
| `swap_n` | 8 |
| `nnls_layer` | 9 |
| `nnls_top5` | 0.125 |
| `nnls_random` | 0.0 |
| `confirm_n` | 24 |
| `confirm_wins_j` | 16 |
| `confirm_wins_logit` | 6 |
| `confirm_sign_p` | 0.052479 |
| `confirm_median_rank_j` | 5.0 |
| `confirm_median_rank_logit` | 8.5 |
| `confirm_top10_j` | 15 |
| `confirm_top10_logit` | 14 |
<!-- /table:summary -->
