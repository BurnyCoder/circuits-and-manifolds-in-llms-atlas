---
slug: number-helix
title: The number helix (trigonometric addition)
category: manifold
collection: tegmark
models: [Pythia-160M, Pythia-410M, Pythia-1B, Pythia-1.4B, Pythia-2.8B]
fidelity: recomputed
evidence: mixed
order: 110
summary: Larger models store numbers on a spiral (a line wound around circles of period 2, 5, 10 and 100) and use it to add like a clock. In Pythia up to 2.8B the helix fit is partial, and patching it beats a random 9-D subspace but not the top principal directions.
viz: number-helix
data: number-helix.json
locations:
  - { model: pythia-160m, part: embed, label: number-token embeddings are periodic (T ≈ 2, 5, 10) }
  - { model: pythia-160m, part: resid, layers: [1, 11], label: generalized helix fit per layer (see entry) }
  - { model: llama-3.1-8b, part: resid, layers: [9, 18], label: helix(a), helix(b) moved and combined into helix(a+b) (Kantamneni & Tegmark; GPT-J analogue) }
tags: [numbers, Fourier features, helix, activation patching, scaling]
related: [grokking, calendar-circles, geometric-calculator, greater-than]
citations:
  - id: kantamneni2025
    title: Language Models Use Trigonometry to Do Addition
    authors: Kantamneni, Tegmark
    year: 2025
    venue: ICLR 2025 Workshop on Reasoning and Planning for LLMs
    url: https://arxiv.org/abs/2502.00873
  - id: zhou2024
    title: Pre-trained Large Language Models Use Fourier Features to Compute Addition
    authors: Zhou, Fu, Sharan, Jia
    year: 2024
    venue: NeurIPS 2024
    url: https://arxiv.org/abs/2406.03445
  - id: biderman2023
    title: "Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling"
    authors: Biderman et al.
    year: 2023
    venue: ICML 2023
    url: https://arxiv.org/abs/2304.01373
replications:
  - who: This atlas
    url: https://github.com/BurnyCoder/circuits-and-manifolds-in-llms-atlas/blob/main/src/atlas/entries/number_helix.py
    model: Pythia 160M → 2.8B
    outcome: ours
    note: First test of the helix below 6B parameters that we know of — embedding spectra, cross-validated helix fits with shuffled controls, and activation patching against PCA-9 and random 9-D subspaces.
  - who: Neuronpedia — The Circuits Research Landscape (attribution graphs)
    url: https://www.neuronpedia.org/graph/info
    model: attribution graphs (model not stated)
    outcome: partial
    note: Inconclusive rather than a failure. Addition graphs show lookup-table features and do not seem to recover the helix; the authors present this as a limitation of transcoders (they may not preserve geometry), not as evidence that the model lacks a helix.
---

## In plain words

How does a language model represent "37"? Kantamneni & Tegmark found that, inside GPT-J and other 6–8B models, a number
is stored like a point on a **spiral staircase**: one coordinate grows steadily with the number (the staircase going
up) while other coordinates go around circles that repeat every 2, 5, 10 and 100 numbers (the ones digit, whether it is
even, and so on). To add, the model rotates these circles like the hands of a clock — the "Clock" algorithm. We test
whether small models already have this spiral.

## How it works (technical)

The paper models the residual stream at number token $a$ as
$\text{helix}(a) = C\,B(a)$ with $B(a) = [a, \cos(2\pi a/T), \sin(2\pi a/T)]_{T \in \{2,5,10,100\}}$, fitted by linear
regression after projecting to 100 PCA dimensions. Patching along the fitted helix recovers a logit difference of 7.21
versus 8.34 for patching the whole layer (PCA: 6.13) in GPT-J; attention heads in layers 9–14 move helix(a) and
helix(b) to the last token, and MLPs 14–18 build helix(a + b).

**What we measured** ({{n_models}} Pythia models; headline numbers for {{largest}}):

- The number-token embeddings' strongest Fourier periods are {{top_periods}}.
- At layer {{best_layer}} the helix predicts held-out numbers' activations with $R^2$ = {{helix_test_r2}}, versus
  {{shuffled_test_r2}} when the same basis is assigned to numbers in random order, {{linear_test_r2}} for the linear
  term alone and {{circles_test_r2}} for circles without the linear term.
- Few-shot addition accuracy (a + b < 100) is {{acc_largest|pct}}. On correctly answered prompts, at layer
  {{patch_layer}}, moving the number's activation along the helix recovers a median {{patch_helix|pct}} of the clean
  logit difference, versus {{patch_full|pct}} for the full layer, {{patch_pca9|pct}} for the top-9 PCA directions and
  {{patch_linear|pct}} for the linear direction alone ({{patch_n}} prompts).
- With 5-fold cross-validation over the numbers and the best layer chosen for every basis (the shuffled control
  included), the helix reaches $R^2$ = {{helix_cv_r2}} (layer {{cv_layer}}) vs {{shuffled_cv_r2}} shuffled: just under the
  pre-stated 0.5, partial as before.
- Confirmatory patch test (pre-stated 2026-11-28), at every layer where the full-layer patch recovers at least half
  the effect: {{patch_spec}} (mean per-prompt recovery; paired differences with 95% bootstrap CIs). The helix carries
  far more than a random 9-D direction set from the same activation span, but no more than the top-9 PCA directions,
  so the patching result supports "a low-dimensional number subspace", not the helix in particular (partial).

## Limitations

- Pythia models are smaller and much worse at arithmetic than the paper's models; we restrict to a + b < 100 so answers
  are single tokens.
- Activations are averaged over three few-shot prefixes; the representation at token $a$ cannot depend on $b$ (causal
  attention).
- We do not reproduce the neuron-level analysis (MLP neurons as sums of cosines).
