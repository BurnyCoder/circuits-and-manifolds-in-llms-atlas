---
slug: belief-fractal
title: Belief-state fractals in the residual stream
category: manifold
collection: other
models: [4-layer transformer (trained here)]
fidelity: recomputed
evidence: correlational
order: 130
summary: A transformer trained on a simple hidden Markov process encodes the Bayes-optimal beliefs — a fractal in a triangle — linearly in its activations.
viz: belief-fractal
data: belief-fractal.json
locations:
  - { model: mess3-4l, part: resid, layer: 3, label: belief-state simplex (fractal) is linearly embedded in the final residual stream }
tags: [computational mechanics, belief states, world models, fractals]
related: [othello-gpt, toy-superposition]
citations:
  - id: shai2024
    title: Transformers Represent Belief State Geometry in their Residual Stream
    authors: Shai, Marzen, Teixeira, Gietelink Oldenziel, Riechers
    year: 2024
    venue: arXiv
    url: https://arxiv.org/abs/2405.15943
replications:
  - who: This atlas
    url: https://github.com/BurnyCoder/circuits-and-manifolds-in-llms-atlas/blob/main/src/atlas/entries/belief_fractal.py
    model: 4-layer transformer trained on Mess3
    outcome: ours
    note: Trained from scratch; linear readout of exact Bayesian beliefs across training, with a random-init control and the optimal-loss bound.
---

## In plain words

Imagine a machine with three hidden moods that emits letters A, B, C, each mood favouring one letter. You only see the
letters. The best possible guess about the hidden mood after a sequence is a set of probabilities — a point inside a
triangle (one corner per mood). As you consider all possible letter histories, these "belief" points form a beautiful
**fractal**. Shai et al. trained a transformer only to predict the next letter and found the same fractal, linearly, in
its activations: the network computes the Bayes-optimal belief even though nobody asked it to.

## How it works (technical)

Mess3 has labelled transition matrices $T^{(t)}_{ij} = P(\text{emit } t, \text{next state } j \mid \text{state } i)$;
an observer updates beliefs as $\eta \leftarrow \eta T^{(t)} / (\eta T^{(t)} \mathbf{1})$. Because the next-token
distribution depends on the history only through $\eta$, an optimal predictor must carry $\eta$; the paper shows an affine
map from the final residual stream recovers it (their Fig. 3) and that for other processes the belief geometry can be
spread across layers.

**What we measured** ({{steps}} training steps):

- A linear map from the final residual stream predicts the true belief with held-out $R^2$ = {{final_r2}} (MSE
  {{final_mse}}); the same readout on the untrained network gives {{rand_r2}}. Concatenating all layers gives
  {{all_layers_r2}}.
- The model's loss is {{final_loss}} nats against an optimal {{optimal_loss}} and a uniform-guess {{uniform_loss}}.

## Limitations

- One small process (Mess3) and one architecture; the paper's more striking result (beliefs spread across layers for
  RRXOR) is not reproduced here.
- A linear readout shows the information is present and linearly accessible, not that the model uses it (correlational).
