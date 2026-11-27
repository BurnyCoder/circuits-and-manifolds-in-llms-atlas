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
| Claim | Published | Ours | Pre-stated criterion | Verdict |
|---|---|---|---|---|
| Days and months are arranged in cyclic order (GPT-2 small, layer 7) | circles for days, months, years in GPT-2 L7 (Fig. 1) | days order 1.00 (p = 0.003); months 1.00 (p = 0.0000); random-word control significant in 2% of 40 sets | both p < 0.05 and control false-positive rate ≤ 10% | ✅ pass |
| SAE latents reconstruct the circle | SAE-cluster reconstruction shows the circle (Fig. 1) | reconstruction from the top 24 day latents: cyclic order 1.00; months 1.00 | cyclic order ≥ 0.7 for both | ✅ pass |
| The circle is causally used for day arithmetic (Qwen3-1.7B) | circular-probe patching shifts the answer in Mistral 7B / Llama 3 8B (§5) | accuracy 68% on 63 prompts; best layer 5: patch success 30% vs 0% for a random 2-D plane (n = 43) | some layer reaches ≥ 50% success with its random-plane control ≤ 20% | 🟡 partial |
<!-- /table:replication -->

## Interpretation

The circular geometry replicates cleanly in GPT-2 small, both in raw activations and through SAE latents. The causal test is partial. Moving the start day along the circle shifts the answer by the matching amount in a minority of correctly solved prompts, while a random plane never does. The circle carries real information the model uses, but a 2-D patch at one position does not control the computation by itself in Qwen3-1.7B.

## Limitations

- The causal test uses our own day-arithmetic templates (see the amendment in `planned-runs.md`), not the paper's Mistral/Llama setups with circular probes.
- Patching happens at one layer and one position at a time.

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
