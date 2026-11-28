---
slug: toy-superposition
title: Toy models of superposition (live)
category: feature
collection: anthropic-tc
models: [toy ReLU autoencoder (in your browser)]
fidelity: recomputed
evidence: causal
order: 5
summary: When features are rare, a network packs more of them than it has dimensions — as a pentagon of directions. Train it yourself, live.
viz: toy-superposition
tags: [superposition, polytopes, sparsity, phase change]
related: [belief-fractal, sae-manifolds]
citations:
  - id: elhage2022
    title: Toy Models of Superposition
    authors: Elhage, Hume, Olsson, Schiefer, … Wattenberg, Olah
    year: 2022
    venue: Transformer Circuits Thread
    url: https://transformer-circuits.pub/2022/toy_model/index.html
  - id: tms-code
    title: toy-models-of-superposition (official notebook)
    authors: Anthropic
    year: 2022
    url: https://github.com/anthropics/toy-models-of-superposition
replications:
  - who: zroe1 — replication of Toy Models of Superposition
    url: https://github.com/zroe1/toy-models-of-superposition
    model: toy ReLU models
    outcome: partial
    note: Superposition and its dependence on sparsity replicate; the trained phase diagrams only roughly match theory (marked as still in progress); results depend strongly on learning rate and batch size.
---

## In plain words

Neural networks seem to know far more concepts than they have neurons. How? Elhage et al. built the smallest possible
test: squeeze 5 features into 2 numbers and ask the network to reconstruct them. If the features are common (often
present together), the network keeps only the two most important ones, each on its own axis. If they are rare, it
discovers a trick: it stores **all five** at once, spread out like the points of a pentagon, and accepts a little
interference when two happen to appear together. That trick is called *superposition*, and it is why single neurons in
real models often respond to several unrelated things — the motivation for sparse autoencoders.

## How it works (technical)

The model is $x' = \text{ReLU}(W^\top W x + b)$ with $W \in \mathbb{R}^{2\times 5}$, trained on sparse inputs (each
feature present with probability $1-S$, value uniform on [0, 1]) with importance-weighted error
$\sum_i I_i (x_i - x'_i)^2$, $I_i = \text{decay}^i$. Elhage et al. measure each feature's *dimensionality*
$D_i = \lVert W_i\rVert^2 / \sum_j (\hat W_i \cdot W_j)^2$: 1 for a dedicated dimension, ½ for antipodal pairs, ⅖ for a
pentagon, 0 for an ignored feature; as sparsity increases the model moves through these states in sharp phase
transitions, and the optimal configurations match uniform polytopes.

This entry trains the model in your browser (Adam, 4,000 steps of 256 samples) every time you move a slider — the
numbers in the panel are computed live, not precomputed.

## Limitations

- A five-feature, two-dimensional toy; real models have millions of features and nonlinear interactions.
- Results depend on random initialization and optimizer settings (try "re-initialize").
