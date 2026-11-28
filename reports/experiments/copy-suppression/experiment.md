# Experiment: copy suppression (GPT-2 small L10H7)

Code [`src/atlas/entries/copy_suppression.py`](../../../src/atlas/entries/copy_suppression.py) · site data
`site/public/data/copy-suppression.json` · page `#/entry/copy-suppression` · fidelity 🟢 recomputed.

## Question

Is head 10.7 a copy-suppression head (McDougall et al. 2023, [arXiv 2310.04625](https://arxiv.org/abs/2310.04625)):
does its OV circuit write negatively onto the token it attends to, and does its QK circuit attend to tokens that are
currently being predicted?

## Setup

- **OV test:** effective embedding MLP0(W_E) of each of 5,000 frequent tokens → head OV → W_U; rank the effect on the
  same token among all output logits (rank 1 = most negative). Statistic: % of tokens whose own logit is among the
  10 most negative, and in the bottom 5%.
- **QK test:** query W_U[t], keys MLP0(W_E[s]) for all s; % of t where s = t gets the highest score.
- **Behaviour:** 8 natural prompts with a repeated token (e.g. "All's fair in love and"); direct effect of L10H7 on the
  copied token's logit; zero-ablation of the head.

**Controls.** The same OV statistic for every other head in layers 10–11.

**Why these settings.** The paper ranks within columns (for each output token, which source tokens suppress it); we
rank within rows (for each attended token, which outputs it suppresses), the more direct reading of "suppresses what it
attends to". The numbers are therefore comparable in kind but not identical; the criterion is relative to the other
late heads to make it robust to this choice.

## Results vs published claims

<!-- table:replication -->
| Claim | Published | Ours | Criterion | Verdict |
|---|---|---|---|---|
| L10H7's OV circuit suppresses the token it attends to | 84.70% of tokens: diagonal among 10 most negative; 98.86% in bottom 5% (column ranking) | 66.6% in top-10 most negative; 94.0% in bottom 5% (row ranking, 5000 tokens); other L10–11 heads: median 0.0% | > 50% top-10 and > 5× the median other head in layers 10–11 | ✅ pass |
| L10H7's QK circuit attends to tokens that are being predicted | 95.72% of rows have the diagonal as maximum | 85.6% diagonal-max over 5000 tokens | > 50% diagonal-max | ✅ pass |
| On natural prompts L10H7 lowers the logit of a copied token | worked examples, e.g. 'All's fair in love and' (Table 1) | negative direct effect on 8/8 prompts; ablation raises the copied token's probability on 5/8 | negative direct effect on ≥ 75% of prompts | ✅ pass |
| Confirmatory: L10H7 suppresses the copied token specifically | copy suppression lowers the logit of tokens that appear earlier in the context (McDougall et al.) | direct effect lower on the copied token than on a control token (most probable next token not in the prompt) on 62% [43, 79] of 24 prompts; negative on 100% [86, 100] (Wilson 95% CIs) | both ≥ 75% of the 24 prompts; partial if one holds (pre-stated 2026-11-28) | 🟡 partial |
<!-- /table:replication -->

## Interpretation

L10H7's OV circuit is strongly self-suppressive and its QK circuit attends to predicted tokens, far above every other
layer-10/11 head. Behaviourally, the head pushes down the copied token on all 8 hand-written prompts (no control token
is measured), and zero-ablation raises that token's probability on only 5 of 8. One hypothesis, not tested here, is
that other components such as negative name mover 11.10 (which the paper also finds partly copy-suppressive)
compensate.

**Confirmatory control (pre-stated 2026-11-28).** On 24 prompts (the 8 above plus 16 written before running), the
direct effect on the copied token is negative in every prompt (`n_neg_direct_all`), but it is lower than the direct
effect on a control token (the most probable next token absent from the prompt) in only `n_below_control` of 24, below
the 75% threshold, so the row is partial. Inspecting the per-prompt values in the site data, many control tokens are
variants of prompt words (" King" / "king", " doors" / "door", " Alice", " ocean" / "sea"), and the head lowers those
too. The data support "L10H7 suppresses tokens related to what it attends to", with the exact copy not singled out on a
quarter to a half of prompts.

## Deviations from the paper

- Row- instead of column-ranking (see above); 5,000 instead of all tokens.

## Limitations

- 24 hand-written prompts; the paper's evaluation on OpenWebText (CSPA) is not reproduced.
- The control token is the model's own top prediction outside the prompt, which is often a near-variant of a prompt
  word; a semantically unrelated control would make the test easier to pass.

## All summary values

<!-- table:summary -->
| Key | Value |
|---|---|
| `n_tokens` | 5000 |
| `ov_top10` | 0.6658 |
| `ov_bottom5` | 0.9396 |
| `qk_diag` | 0.856 |
| `ctrl_top10_median` | 0.0 |
| `ctrl_top10_max` | 0.4012 |
| `ctrl_top10_argmax` | 11.10 |
| `n_prompts` | 8 |
| `n_prompts_all` | 24 |
| `n_neg_direct` | 8 |
| `n_ablation_up` | 5 |
| `n_neg_direct_all` | 24 |
| `n_ablation_up_all` | 13 |
| `mean_attn_copy` | 0.335905 |
| `n_below_control` | 15 |
<!-- /table:summary -->
