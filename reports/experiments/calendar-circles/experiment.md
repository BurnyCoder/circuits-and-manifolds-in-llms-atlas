# Experiment: Calendar circles (days and months)

Code [`src/atlas/entries/calendar_circles.py`](../../../src/atlas/entries/calendar_circles.py) (the docstring gives the full method) · site
data `site/public/data/calendar-circles.json` · page `#/entry/calendar-circles` · pre-stated criteria and dated amendments:
[`reports/planned-runs.md`](../../planned-runs.md).

## Question

Are days of the week and months arranged on circles in language-model activations (Engels et al. 2024, [arXiv 2405.14860](https://arxiv.org/abs/2405.14860)), do SAE latents tile those circles, and is the circle causally used for day arithmetic?

## Setup

GPT-2 small layer 7 residual activations at day/month tokens, with the template mean removed, are projected on their top PCs. Cyclic order is scored with a permutation null and compared with random word sets. The circle is also reconstructed from the top `gpt2-small-res-jb` SAE latents for days. For Qwen3-1.7B, layers 2–20 are swept. The causal test shifts the start-day token's 2-D circle component from day d's mean to day d + j's mean, on day-arithmetic prompts the model solves, and checks whether the answer shifts by j. The control is a random 2-D plane.

## Results vs published claims

<!-- table:replication -->
| Claim | Published | Ours | Criterion | Verdict |
|---|---|---|---|---|
| Days and months are arranged in cyclic order (GPT-2 small, layer 7) | circles for days, months, years in GPT-2 L7 (Fig. 1) | days order 1.00 (p = 0.003); months 1.00 (p = 0.0000); random-word control significant in 2% of 40 sets (93-word pool); class-mean variance in 2 PCs: days 0.68, months 0.42 vs random-set 95th pct 0.47 (7 words), 0.33 (12 words) | both p < 0.05 and control false-positive rate ≤ 10% | ✅ pass |
| SAE latents reconstruct the circle | SAE-cluster reconstruction shows the circle (Fig. 1) | reconstruction from the top 24 day latents: cyclic order 1.00; months 1.00 | cyclic order ≥ 0.7 for both | ✅ pass |
| The circle is causally used for day arithmetic (Qwen3-1.7B) | circular-probe patching shifts the answer in Mistral 7B / Llama 3 8B (§5) | accuracy 68% on 63 prompts; best layer 5: patch success 30% [19, 45] vs 0% [0, 8] for a norm-matched random 2-D plane (n = 43; Wilson 95% CIs) | some layer reaches ≥ 50% success with its random-plane control ≤ 20% | 🟡 partial (post-hoc: day-arithmetic templates changed after the smoke test showed this test failing) |
<!-- /table:replication -->

## Interpretation

The circular geometry replicates cleanly in GPT-2 small, both in raw activations and through SAE latents; these are
descriptive results (the SAE latents are selected on the same tokens). Since the review (m3) the random-word control
draws its 40 sets from a pool of 93 single-token nouns instead of 20 (`ctrl_sig_rate` of sets still look cyclic at
p < 0.05), and a tied-angle configuration now scores NaN instead of a perfect 1.0. A geometric comparison agrees: days
and months put more of their class-mean variance into two principal components (`g2_days_var2`, `g2_months_var2`) than
the 95th percentile of random word sets of the same size (`ctrl_var2_p95_7`, `ctrl_var2_p95_12`). The causal test is partial and post-hoc (its day-arithmetic templates changed after the smoke test). Moving the
start day along the circle shifts the answer by the matching amount in a minority of correctly solved prompts. Until the
methods review, the random-plane control moved the activation 1–2 orders of magnitude less than the circle patch; it is
now norm-matched and still never succeeds, so the effect is specific to the circle plane, with a wide Wilson interval
(n = 43). A 2-D patch at one position does not control the computation by itself in Qwen3-1.7B, and the best layer is
chosen on the outcome. Details: [methods review](../../reviews/methods-review.md), M4 and m3.

## Limitations

- The causal test uses our own day-arithmetic templates (see the amendment in `planned-runs.md`), not the paper's Mistral/Llama setups with circular probes.
- Patching happens at one layer and one position at a time.
- To do: a "rotate by the wrong amount" control and a fixed layer for the causal test.

## All summary values

<!-- table:summary -->
| Key | Value |
|---|---|
| `g2_days_score` | 1.0 |
| `g2_days_p` | 0.0026 |
| `g2_months_score` | 1.0 |
| `g2_months_p` | 0.0 |
| `g2_sae_days_score` | 1.0 |
| `g2_sae_months_score` | 1.0 |
| `g2_sae_days_n` | 24 |
| `ctrl_sig_rate` | 0.025 |
| `g2_days_var2` | 0.676757 |
| `g2_months_var2` | 0.419182 |
| `ctrl_var2_p95_7` | 0.466775 |
| `ctrl_var2_p95_12` | 0.332698 |
| `n_ctrl` | 40 |
| `q_days_best_score` | 1.0 |
| `q_days_best_layer` | 5 |
| `q_months_best_score` | 1.0 |
| `q_months_best_layer` | 8 |
| `q_acc` | 0.68254 |
| `q_causal_best` | 0.302326 |
| `q_causal_best_layer` | 5 |
| `q_causal_best_rand` | 0.0 |
| `q_causal_n` | 43 |
<!-- /table:summary -->
