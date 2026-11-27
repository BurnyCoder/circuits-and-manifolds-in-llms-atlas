# Experiment: Meandering on manifolds (stories in space)

Code [`src/atlas/entries/stories_in_space.py`](../../../src/atlas/entries/stories_in_space.py) (the docstring gives the full method) · site
data `site/public/data/stories-in-space.json` · page `#/entry/stories-in-space` · pre-stated criteria and dated amendments:
[`reports/planned-runs.md`](../../planned-runs.md).

## Question

Is a model's running emotional reading of a story linearly decodable from its activations, and do behaviour and activation geometry agree (Bigelow et al. 2026, [arXiv 2605.12412](https://arxiv.org/abs/2605.12412))?

## Setup

Qwen2.5-1.5B-Instruct reads 160 SimpleStories sentence by sentence (all prefixes batched). After each prefix it rates six emotions (expected 0–9 rating from digit probabilities). Ridge probes per layer, held out by story (α chosen on held-out training stories). Behaviour–activation agreement as RSA over 600 prefixes, with a story-shuffle null.

## Results vs published claims

<!-- table:replication -->
| Claim | Published | Ours | Pre-stated criterion | Verdict |
|---|---|---|---|---|
| The model's running emotional reading is linearly decodable | ridge probes predict per-sentence emotion ratings (RMSE 0.09) from layer-9 activations (Llama-3.1-8B-Instruct) | held-out-story R² 0.54 at layer 23 (1681 prefixes from 160 stories) | R² ≥ 0.5 | ✅ pass |
| Behaviour and activation geometry agree | distance-matrix correlation r = .92 (emotions) | RSA Spearman r = 0.11 over 600 prefixes (story-shuffle null 95th percentile 0.06) | r ≥ 0.6 (amendment 2026-11-27: computed over prefixes, not 3–6 emotion centroids) | ❌ fail |
<!-- /table:replication -->

## Interpretation

The emotional reading is linearly decodable from activations of unseen stories, best in late layers (pass). Behaviour and activation geometry agree only weakly: the RSA correlation is above the story-shuffle null but far below the paper's value (fail). The model's internal geometry carries the ratings, but the full prefix-by-prefix distance structure is not shared.

## Limitations

- SimpleStories are mostly cheerful, so ratings vary little for some emotions.
- Expected digit ratings from a 1.5B model are a noisy behavioural readout.

## All summary values

<!-- table:summary -->
| Key | Value |
|---|---|
| `n_stories` | 160 |
| `n_prefixes` | 1681 |
| `best_layer` | 23 |
| `best_r2` | 0.542713 |
| `r_geom` | 0.111546 |
| `r_geom_null95` | 0.060774 |
| `n_rsa_prefixes` | 600 |
<!-- /table:summary -->
