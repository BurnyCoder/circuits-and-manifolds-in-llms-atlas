# Experiment: The refusal direction

Code [`src/atlas/entries/refusal_direction.py`](../../../src/atlas/entries/refusal_direction.py) (the docstring gives the full method) · site
data `site/public/data/refusal-direction.json` · page `#/entry/refusal-direction` · pre-stated criteria and dated amendments:
[`reports/planned-runs.md`](../../planned-runs.md).

## Question

Is refusal mediated by a single direction in an instruction-tuned model (Arditi et al. 2024, [arXiv 2406.11717](https://arxiv.org/abs/2406.11717))?

## Setup

Qwen2.5-1.5B-Instruct; difference-of-means direction between harmful and harmless instructions (the paper's splits), layer selected on validation prompts; directional ablation at all layers on held-out harmful prompts, addition on held-out harmless prompts; controls with a norm-matched random direction; refusal judged by refusal-phrase matching.

## Results vs published claims

<!-- table:replication -->
| Claim | Published | Ours | Criterion | Verdict |
|---|---|---|---|---|
| Ablating one direction removes refusal | directional ablation sharply lowers refusal on harmful prompts in 13 chat models (Arditi et al. Fig. 1) | refusal on 64 harmful test prompts: 86% [75, 92] → 0% [0, 6] (random direction: 84% [74, 91]; Wilson 95% CIs) | ablation cuts refusal by ≥ 50 points while the random direction changes it by < 15 | ✅ pass |
| Adding the direction makes the model refuse harmless requests | activation addition induces refusal on harmless instructions (§3) | refusal on 64 harmless prompts: 6% [2, 15] → 98% [92, 100] (random direction: 6% [2, 15]; Wilson 95% CIs) | addition raises refusal by ≥ 50 points while the random direction raises it by < 15 | ✅ pass |
| Control: a label-shuffled direction neither removes nor induces refusal | the refusal effect is specific to the harmful − harmless direction | shuffled-label direction: harmful refusal 86% [75, 92] → 94% [85, 98] when ablated; harmless refusal 6% [2, 15] → 2% [0, 8] when added (Wilson 95% CIs) | both changes < 15 points | ✅ pass (post-hoc: stronger control added after the methods review) |
<!-- /table:replication -->

## Interpretation

Both claims replicate. Ablating the direction removes refusal on harmful prompts, and adding it makes the model refuse
harmless requests. The original ablation control (a random unit direction) could hardly fail, because projecting out a
random direction removes about 1/d of the activation variance; the label-shuffled difference-of-means direction added
after the methods review is a real activation direction, and it leaves refusal near baseline in both tests, so the
effect is specific to the harmful-vs-harmless contrast. "Refusal removed" is still judged by the absence of refusal
phrases only; incoherent output would count as compliance. Details: [methods review](../../reviews/methods-review.md), M14.

## Limitations

- Substring-based refusal detection; no judge model and no harmfulness scoring of the completions (by design, they are not stored).
- One model size.
- To do: a coherence check that stores no text (perplexity under the unablated model, or a classifier label).

## All summary values

<!-- table:summary -->
| Key | Value |
|---|---|
| `layer` | 20 |
| `n_layers` | 28 |
| `harm_base` | 0.859375 |
| `harm_ablate` | 0.0 |
| `harm_rand` | 0.84375 |
| `safe_base` | 0.0625 |
| `safe_add` | 0.984375 |
| `safe_rand` | 0.0625 |
| `harm_shuf` | 0.9375 |
| `safe_shuf` | 0.015625 |
| `n_test` | 64 |
<!-- /table:summary -->
