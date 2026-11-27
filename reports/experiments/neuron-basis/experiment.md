# Experiment: Neurons as an interpretable basis (Goodfire)

Code [`src/atlas/entries/neuron_basis.py`](../../../src/atlas/entries/neuron_basis.py) (the docstring gives the full method) · site
data `site/public/data/neuron-basis.json` · page `#/entry/neuron-basis` · pre-stated criteria and dated amendments:
[`reports/planned-runs.md`](../../planned-runs.md).

## Question

How sparse are SwiGLU neuron activations per token in a dense model: can most be zeroed with little damage, as Goodfire found in an MoE model, and do gate-open quadrants carry the causal weight?

## Setup

Qwen3-1.7B (28 layers × 6,144 neurons) on 48 × 512 WikiText-103 tokens. Per-token top-k by |activation| before the down projection in every layer, rest zeroed (Goodfire's layer top-k rule), for k from 3,072 down to 64; KL and loss vs the intact model, recomputed per sequence. Controls: random-480, zeroing only the top 480, P25–P75 band mean-ablation. Post hoc: ericzhou12's rule (top-k by distance from mean, mean-filled). Quadrant census and gradient × activation attribution.

## Results vs published claims

<!-- table:replication -->
| Claim | Published | Ours | Pre-stated criterion | Verdict |
|---|---|---|---|---|
| Most neuron activations can be zeroed per token with little damage | keep 320/4,096 (92.2% ablated): loss 1.886 → 1.940 (+2.9%), KL 0.140 (Qwen3.6-35B-A3B MoE); open Qwen3-1.7B replication: KL ≈ 0.80 | keep 480/6,144 (92.2% ablated): loss 3.030 → 4.112 (+35.7%), KL 1.609; random-480 → 14.910 | relative loss increase < 25% and KL < 1.0, and the random-480 control at least 3× worse in loss increase | 🟡 partial |
| The few large activations carry the computation | ablating only the top 12.5% per expert → loss 4.587 (bottom-448 control) | zeroing only the top 480 per token → loss 14.329 (base 3.030) | loss increase ≥ 5× that of keeping only the top 480 | ✅ pass |
| Gate-open quadrants carry causal weight although most activations are gate-closed | top attributed cells: Q1 39–50% + Q2 43–57%; 71–72% of all activations in Q3/Q4 | top-1% gradient×activation cells: 56% gate-open (Q1+Q2); all activations: 80% gate-closed | ≥ 80% of top cells gate-open and ≥ 50% of all activations gate-closed | ❌ fail |
<!-- /table:replication -->

## Interpretation

In this dense model, Goodfire's rule at 92.2% sparsity costs much more than in their MoE: the claim is partial, since random neurons are far worse but the KL is above the threshold. ericzhou12's gentler rule gives a KL close to their reported value, so the gap with that replication comes from the selection rule. Zeroing only the strongest activations is catastrophic, confirming that the few large activations carry the computation. Most activations are gate-closed, as Goodfire found, but the top attributed cells are only modestly gate-open (fail).

## Limitations

- Dense 1.7B model vs Goodfire's 35B MoE; their router weighting has no analogue here.
- WikiText only; no benchmark evaluation or recovery fine-tuning.

## All summary values

<!-- table:summary -->
| Key | Value |
|---|---|
| `base_loss` | 3.030207 |
| `k480_loss` | 4.112269 |
| `k480_rel` | 0.357092 |
| `k480_kl` | 1.609199 |
| `random480_loss` | 14.9095 |
| `drop480_loss` | 14.328905 |
| `band_loss` | 3.045164 |
| `band_kl` | 0.052616 |
| `k480_meanrule_kl` | 0.85285 |
| `k480_meanrule_loss` | 3.498474 |
| `gate_closed_frac` | 0.803565 |
| `top_attr_gate_open` | 0.556747 |
| `n_tokens` | 24528 |
<!-- /table:summary -->
