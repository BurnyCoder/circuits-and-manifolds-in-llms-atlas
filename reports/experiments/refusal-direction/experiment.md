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
| Claim | Published | Ours | Pre-stated criterion | Verdict |
|---|---|---|---|---|
| Ablating one direction removes refusal | directional ablation sharply lowers refusal on harmful prompts in 13 chat models (Arditi et al. Fig. 1) | refusal on 64 harmful test prompts: 86% → 0% (random direction: 84%) | ablation cuts refusal by ≥ 50 points while the random direction changes it by < 15 | ✅ pass |
| Adding the direction makes the model refuse harmless requests | activation addition induces refusal on harmless instructions (§3) | refusal on 64 harmless prompts: 6% → 98% (random direction: 6%) | addition raises refusal by ≥ 50 points while the random direction raises it by < 15 | ✅ pass |
<!-- /table:replication -->

## Interpretation

Both claims replicate strongly. Ablating the direction removes refusal on harmful prompts while a random direction leaves it intact. Adding the direction makes the model refuse harmless requests, which a random direction does not do.

## Limitations

- Substring-based refusal detection; no judge model and no harmfulness scoring of the completions (by design, they are not stored).
- One model size.

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
| `n_test` | 64 |
<!-- /table:summary -->
