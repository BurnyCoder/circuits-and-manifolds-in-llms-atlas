# Experiment: indirect object identification (IOI)

Code [`src/atlas/entries/ioi.py`](../../../src/atlas/entries/ioi.py) · site data `site/public/data/ioi.json` · page
`#/entry/ioi` · fidelity 🟢 recomputed.

## Question

Does GPT-2 small solve IOI ("When Mary and John went to the store, John gave a drink to" → " Mary") with the 26-head,
7-class circuit of Wang et al. 2022 ([arXiv 2211.00593](https://arxiv.org/abs/2211.00593)), and is that circuit faithful
and necessary?

## Setup

- **Model:** GPT-2 small via TransformerLens 3.9.
- **Data:** 8 templates × 40 prompts = 320 prompts in the paper's style (BABA and ABBA name orders), names / places /
  objects filtered to single GPT-2 tokens so positions align within a template; the corrupted p_ABC set replaces all
  three names with different names.
- **Metric:** logit difference logit(IO) − logit(S) at the END token, with a 95% bootstrap CI over prompts.
- **Per-head evidence:** direct logit attribution onto W_U[IO] − W_U[S]; noising (replace one head's output with its
  p_ABC output); name-mover attention from END to IO vs S.
- **Knockouts:** mean-ablation (per template, per position, mean over p_ABC) of each of the 2⁷ = 128 combinations of
  the 7 head classes; faithfulness = keep only the 26 circuit heads, mean-ablate the other 118.

**Controls.** 40 random sets of 26 heads, mean-ablated the same way.

**Why these settings.** Mean-ablation over p_ABC follows the paper's §4 knockouts; 320 prompts give a logit-diff CI of
about ±0.17; 128 combinations are cheap (one forward pass each over 320 prompts) and give the toggleable DAG on the site.

## Results vs published claims

<!-- table:replication -->
| Claim | Published | Ours | Criterion | Verdict |
|---|---|---|---|---|
| GPT-2 small prefers the indirect object over the subject | mean logit diff 3.56; IO > S in 99.3% of prompts | 3.54 [3.37, 3.70] on 320 prompts; IO > S in 99.4% | mean logit diff > 2 and IO > S in > 90% of prompts | ✅ pass |
| Name movers (9.9, 9.6, 10.0) attend from END to the indirect object | mean attention to IO 0.59 (§3.1) | mean attention to IO 0.60 vs to S 0.09 | attention to IO > 2× attention to S | ✅ pass |
| Negative name movers (10.7, 11.10) write against the answer | negative copy score 98% (§3.4) | direct logit attribution -2.11 and -0.92 | both heads have negative DLA on logit(IO) − logit(S) | ✅ pass |
| The 26-head circuit is faithful | circuit keeps 87% of the logit difference (§4) | circuit-only logit diff 3.29 [3.12, 3.46] = 93% of full model | between 60% and 120% of the full model | ✅ pass |
| Knocking out the circuit breaks the task; random heads don't | circuit knockouts collapse performance; random heads matter little (§4) | all 7 classes ablated → -1.27; 26 random heads → median 2.48 | circuit knockout drops logit diff by > 70%, random median drops it by < 40% | ✅ pass |
<!-- /table:replication -->

## Interpretation

All five claims replicate, with the baseline numbers within sampling error of the paper (our prompts differ from
theirs). The circuit keeps more of the logit difference than the paper reports (93% vs 87%). Part of that is
leniency: we keep circuit heads clean at **all** positions, whereas Wang et al. define circuit nodes as (head, token
position) pairs (§2.1), e.g. name movers at END only. Simpler, more uniform templates may also contribute. Knocking
out all 7 classes reverses the preference, while 26 random heads leave most of it intact.

## Deviations from the paper

- Own templates/names (the paper's exact dataset is not used), 8 instead of 15 templates.

## Limitations

- Faithfulness is measured by one metric (logit diff) with one ablation type; Miller et al. 2024
  ([arXiv 2407.08734](https://arxiv.org/abs/2407.08734)) show faithfulness numbers vary strongly with ablation choice.
- To do: mean-ablate circuit heads outside their class positions (S1+1, S2, END).

## All summary values

<!-- table:summary -->
| Key | Value |
|---|---|
| `n_prompts` | 320 |
| `ld_clean` | 3.537183 |
| `ld_clean_lo` | 3.373421 |
| `ld_clean_hi` | 3.701031 |
| `ld_abc` | 0.01084 |
| `io_win` | 0.99375 |
| `faithful_ld` | 3.289322 |
| `faithful_frac` | 0.929927 |
| `all_classes_ablated_ld` | -1.270267 |
| `rand_ablated_ld_median` | 2.475566 |
| `nm_attn_io` | 0.598112 |
| `nm_attn_s` | 0.090859 |
| `neg_nm_dla_10_7` | -2.109669 |
| `neg_nm_dla_11_10` | -0.919202 |
| `top_dla_head` | 9.9 |
<!-- /table:summary -->
