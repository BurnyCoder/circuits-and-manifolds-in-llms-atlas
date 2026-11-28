# Experiment: The character-count manifold (open model)

Code [`src/atlas/entries/linebreak_manifold.py`](../../../src/atlas/entries/linebreak_manifold.py) (the docstring gives the full method) · site
data `site/public/data/linebreak-manifold.json` · page `#/entry/linebreak-manifold` · pre-stated criteria and dated amendments:
[`reports/planned-runs.md`](../../planned-runs.md).

## Question

Is the number of characters since the last line break linearly decodable, low-dimensional and curved in open models, as Gurnee et al. 2025 found in Claude 3.5 Haiku ([Transformer Circuits](https://transformer-circuits.pub/2025/linebreaks/index.html))?

## Setup

WikiText-103 paragraphs wrapped at widths 40–150 characters. For every token, the characters since the last newline. Qwen3-1.7B (and GPT-2 small as a weaker comparison): per-layer ridge probes on held-out documents (α chosen on held-out training documents) vs shuffled labels; PCA of per-count means (variance in 6 PCs, turning angle); the newline-prediction AUC. Position 0 of each document is excluded (attention sink).

## Results vs published claims

<!-- table:replication -->
| Claim | Published | Ours | Criterion | Verdict |
|---|---|---|---|---|
| Character count since the last newline is linearly decodable | Haiku: probe R² 0.985 after layer 1; open models (T-Tech): Qwen3-8B 0.852 at L5, Gemma-2-9B 0.868 | Qwen3-1.7B: best held-out R² 0.747 at layer 6 (shuffled labels -0.005); GPT-2 small 0.759 | Qwen3-1.7B best R² ≥ 0.7 and ≥ 0.5 above the shuffled control | ✅ pass |
| The counts lie on a low-dimensional curved manifold | Haiku: top 6 PCs of per-count means hold 95% of variance; helix-like curve with ringing | Qwen3-1.7B layer 6: 6 PCs hold 66% (3 PCs 49%); PC1–2 angle turns 1.93°/char | 6 PCs ≥ 80% of variance of the per-count means | ❌ fail |
| The count is represented in the very first layers | Haiku: R² 0.985 already after layer 1 (T-Tech: early peak not replicated in open models) | Qwen3-1.7B: best R² within layers 0–2 = 0.668 (best overall 0.747 at layer 6) | early-layer R² ≥ 0.9 × the best layer's R² | ❌ fail |
| The model breaks lines where the text does | Haiku predicts newlines at the wrap width (newline probability 0.79 in their example) | AUC of P(newline) for true line breaks: Qwen3-1.7B 0.936, GPT-2 small 0.742 | Qwen3-1.7B AUC ≥ 0.9 | ✅ pass |
<!-- /table:replication -->

## Interpretation

The count is linearly decodable in both models, well above the shuffled control, and both models put high probability
on newlines exactly where the text breaks. Decodability is not specific to the stronger model: GPT-2 small decodes the
count at least as well as Qwen3-1.7B. The margins are thin: claim 1 passes with best-layer R² a few hundredths above
0.70 on an erratic layer profile, and claim 3 fails by less than 0.01, so both are within noise and the verdict layer
is the argmax of test R². Claim 2 fails clearly: the per-count means are not as compact as Claude's (6 PCs hold much
less than 95%). The steady rotation of the leading PCs is consistent with the open replication's helix (T-Tech, period
~155 characters), which also did not find Claude's very-early peak. Details: [methods review](../../reviews/methods-review.md), M9.

## Limitations

- WikiText wrapping is synthetic; the model was not trained to wrap at a fixed width.
- Geometry is measured on class means, which mixes in other token features.
- No CIs; to do: choose the layer on validation documents and bootstrap over documents.

## All summary values

<!-- table:summary -->
| Key | Value |
|---|---|
| `q_best_layer` | 6 |
| `q_best_r2` | 0.747208 |
| `q_best_r2_shuffled` | -0.004575 |
| `q_early_r2` | 0.668107 |
| `q_var6` | 0.656134 |
| `q_var3` | 0.48852 |
| `q_deg_per_char` | 1.933718 |
| `q_period_chars` | 186.1699 |
| `q_auc` | 0.935895 |
| `g_best_r2` | 0.75924 |
| `g_auc` | 0.742246 |
| `q_tokens` | 86607 |
<!-- /table:summary -->
