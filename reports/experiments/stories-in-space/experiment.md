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
| Claim | Published | Ours | Criterion | Verdict |
|---|---|---|---|---|
| The model's running emotional reading is linearly decodable | ridge probes predict per-sentence emotion ratings (RMSE 0.09) from layer-9 activations (Llama-3.1-8B-Instruct) | held-out-story R² 0.54 at layer 23 (1681 prefixes from 160 stories) | R² ≥ 0.5 | ✅ pass |
| Behaviour and activation geometry agree | distance-matrix correlation r = .92 (emotions) | RSA Spearman r = 0.11 over 600 prefixes (story-shuffle null 95th percentile 0.06) | r ≥ 0.6 (amendment 2026-11-27: computed over prefixes, not 3–6 emotion centroids) | ❌ fail (post-hoc: RSA replaced the centroid analysis after the smoke test) |
| Confirmatory: the running emotional reading is linearly decodable (fresh stories) | ridge probes predict per-sentence emotion ratings (RMSE 0.09) from layer-9 activations (Llama-3.1-8B-Instruct) | 200 fresh stories (40 test): R² 0.53 [0.46, 0.57] at validation-chosen layer 17; bag-of-words baseline on the prefix text 0.32 | CI rule at 0.5 (pre-stated 2026-11-28) | 🟡 partial |
| Confirmatory: behaviour and activation geometry agree (fresh stories) | distance-matrix correlation r = .92 (emotions) | RSA Spearman r = 0.13 at layer 17 over 600 prefixes (story-shuffle null 95th percentile 0.07) | r ≥ 0.6 and above the null's 95th percentile; partial if r ≥ 0.3 and above it (pre-stated 2026-11-28) | ❌ fail |
<!-- /table:replication -->

## Interpretation

The emotional reading is decodable from activations of unseen stories, but only just: the best layer passes R² 0.5 by
a few hundredths, only a third of layers reach it, the layer is the argmax of test R², and there is no text baseline
(e.g. a bag-of-words ridge on the prefix). Behaviour and activation geometry agree only weakly: the RSA correlation
(post-hoc, it replaced the centroid analysis after the smoke test) is above the story-shuffle null but far below the
paper's value. Details: [methods review](../../reviews/methods-review.md), M9.

**Confirmatory re-tests (pre-stated 2026-11-28).** 200 fresh SimpleStories (the next qualifying stories after the main
run's 160), the layer chosen on validation stories, and a bootstrap over the 40 test stories.
- *Decodability: partial (inconclusive).* The CI (`f_r2_lo`–`f_r2_hi`) straddles 0.5. A TF-IDF bag-of-words ridge on
  the prefix text reaches only `f_bow_r2`, so the residual stream carries the model's running reading beyond the
  surface words, but how much is not pinned to the 0.5 bar.
- *Geometry: fail.* RSA r (`f_rsa`) is above the story-shuffle null (`f_rsa_null95`), as in the main run, and far below
  the 0.3 partial bar. Behaviour and activation geometry are related, but weakly, in a 1.5B model on cheerful stories.

## Limitations

- SimpleStories are mostly cheerful, so ratings vary little for some emotions.
- Expected digit ratings from a 1.5B model are a noisy behavioural readout.
- The original rows keep their argmax-of-test-R² layer; the confirmatory rows use fresh stories and validation layers.

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
| `f_n_stories` | 200 |
| `f_layer` | 17 |
| `f_r2` | 0.533051 |
| `f_r2_lo` | 0.463497 |
| `f_r2_hi` | 0.565558 |
| `f_bow_r2` | 0.321959 |
| `f_rsa` | 0.125957 |
| `f_rsa_null95` | 0.066349 |
<!-- /table:summary -->
