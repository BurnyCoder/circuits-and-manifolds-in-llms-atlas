---
slug: neuron-basis
title: Neurons as an interpretable basis (and "already clean")
category: neuron
collection: goodfire-neurons
models: [Qwen3-1.7B]
fidelity: recomputed
evidence: causal
order: 300
summary: Each token needs only a small fraction of a model's MLP neurons — zero the rest and the model barely notices; the important ones are "gate-open".
viz: neuron-basis
data: neuron-basis.json
locations:
  - { model: qwen3-1.7b, part: mlp, layer: 0, label: per-token top-k neuron sparsity tested in every MLP layer }
  - { model: qwen3-1.7b, part: mlp, layer: 14, label: SwiGLU sign quadrants (gate-open cells carry the causal weight) }
  - { model: qwen3-1.7b, part: mlp, layer: 27, label: per-token top-k neuron sparsity tested in every MLP layer }
tags: [neurons, SwiGLU, sparsity, monosemanticity]
related: [sae-features, curve-detectors, goodfire-neurons-claims]
quote:
  text: We find that neurons are sparse, concisely describable, and predictable units of computation.
  source: tigges2026
citations:
  - id: tigges2026
    title: Neurons as an Interpretable Basis
    authors: Tigges, Kowal, McGrath, Lewis, Cammarata (Goodfire)
    year: 2026
    url: https://www.goodfire.com/research/neurons-as-an-interpretable-basis
  - id: cammarata2026
    title: Neural Networks Are Already Clean
    authors: Cammarata (Goodfire)
    year: 2026
    url: https://already-clean.goodfire.com/
  - id: arora2026
    title: Language Model Circuits Are Sparse in the Neuron Basis
    authors: Arora, Wu, Steinhardt, Schwettmann
    year: 2026
    venue: ICML 2026
    url: https://arxiv.org/abs/2601.22594
  - id: shazeer2020
    title: GLU Variants Improve Transformer (SwiGLU)
    authors: Shazeer
    year: 2020
    url: https://arxiv.org/abs/2002.05202
replications:
  - who: This atlas
    url: https://github.com/BurnyCoder/circuits-and-manifolds-in-llms-atlas/blob/main/src/atlas/entries/neuron_basis.py
    model: Qwen3-1.7B (dense SwiGLU)
    outcome: ours
    note: Loss/KL-vs-k curve, random and drop-top controls, band ablation, quadrant census and attribution.
  - who: ericzhou12 — neuron-quadrant-interventions
    url: https://github.com/ericzhou12/neuron-quadrant-interventions
    model: Qwen3-1.7B, Qwen3-0.6B
    outcome: partial
    note: Same knee shape but ~6× the damage of the MoE result (KL ≈ 0.80 at 92.2%, different selection rule); zeroing all of Q3 breaks the model.
  - who: alexandertsai — quantized-interpretable-neurons
    url: https://github.com/alexandertsai/quantized-interpretable-neurons
    model: Qwen3.6-35B-A3B (the paper's model)
    outcome: replicated
    note: Top-320 keeps loss within +3.89% (random-320 ≈ +135%); a blind detection test with the released descriptions scored 62% (different protocol).
---

## In plain words

Goodfire argue that individual neurons are much more interpretable than the "superposition" story suggests. One key
measurement: for each token, keep only the few hundred neurons with the largest activations in each layer and set the
other ~92% to zero — the model's predictions hardly change. So each token is processed by a small, sparse set of
neurons, which makes describing what they do feasible. We repeat this on an open model, Qwen3-1.7B.

## How it works (technical)

A SwiGLU MLP computes $a = \text{SiLU}(w_g^\top x)\,(w_u^\top x)$ for every neuron and writes $\sum_n a_n W_{\text{down}}[:, n]$.
Goodfire rank $|a|$ per token across the routed neurons of each layer of Qwen3.6-35B-A3B and keep the top 320 of 4,096
(92.2% zeroed): web-text loss 1.886 → 1.940 (KL 0.140), while keeping a random 64 of 512 per expert gives 4.214. They
also split activations by the signs of gate and up (Q1 g>0,u>0; Q2 g>0,u<0; Q3 g<0,u<0; Q4 g<0,u>0): 71–72% of activations
are gate-closed (Q3/Q4), yet the most causally important cells are gate-open (Q1/Q2).

**What we measured** (Qwen3-1.7B, 28 layers × 6,144 neurons, {{n_tokens}} WikiText-103 tokens):

- Keeping the top 480 neurons per token per layer (92.2% zeroed) changes the loss from {{base_loss}} to {{k480_loss}}
  ({{k480_rel|pct}} relative), KL {{k480_kl}} nats. Keeping a random 480 instead gives {{random480_loss}}; zeroing only
  the top 480 gives {{drop480_loss}}.
- Mean-ablating the central P25–P75 band of every neuron gives {{band_loss}} (KL {{band_kl}}).
- {{gate_closed_frac|pct}} of all activations are gate-closed, but {{top_attr_gate_open|pct}} of the top-1% cells by
  gradient × activation are gate-open.

## Limitations

- Dense 1.7B model vs the paper's 35B mixture-of-experts: in the MoE the router already pre-selects neurons, so 320/4,096
  routed neurons is only 0.24% of a layer's neurons — the dense comparison is not like-for-like.
- We do not reproduce the description, detection, simulation or input-synthesis evaluations (they use large LLM pipelines).
- Critiques worth reading: Neel Nanda's comment on the "already clean" essay (superposition is still real; weak
  activations are not fully addressed) and the question whether +2.9% loss counts as "functionally sparse".
