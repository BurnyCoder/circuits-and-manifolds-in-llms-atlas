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
| Claim | Published | Ours | Pre-stated criterion | Verdict |
|---|---|---|---|---|
| The J-lens surfaces unspoken intermediate steps better than the logit lens | J-lens beats logit / tuned lens; intermediates (21, 42) become readable in middle layers (Claude) | best rank of the intermediate token at the last position: median 4 (J-lens) vs 9 (logit lens); top-10 for 3/3 vs 2/3 | J-lens median rank lower than the logit lens's and ≥ half of intermediates reach the top 10 | ✅ pass |
| Swapping J-lens coordinates swaps the concept | France ↔ China swaps change capital/language answers; 59% top-5 success for the J-space component (Claude) | layer 11: target answer in top-5 for 12% of 8 prompts (random direction 0%) | ≥ 50% and ≥ 30 points above the random-direction swap | ❌ fail |
| Swapping J-space coefficients (paper-style sparse NNLS decomposition) swaps the concept | J-space component swap: 59% top-5 success vs 5% for the non-J component (Claude) | layer 9: target answer in top-5 for 12% of 8 prompts (random vector 0%) | ≥ 50% and ≥ 30 points above the random-vector swap (amendment 2026-11-27, same thresholds) | ❌ fail |
<!-- /table:replication -->

## Interpretation

The lens claim replicates: intermediates rank higher under the J-lens than under the logit lens. Both swap variants fail: swapping France ↔ China–style concepts rarely moves the answer, though always more than random. This matches the open replication avalur/JLens, where selectivity did not reproduce up to 9B, and it suggests the workspace's causal role may need larger models.

## Limitations

- Three rankable intermediate cases and eight swap prompts are small samples.
- Qwen3's digit-level tokenization rules out the paper's arithmetic example.

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
<!-- /table:summary -->
