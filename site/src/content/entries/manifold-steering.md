---
slug: manifold-steering
title: Steering along manifolds
category: manifold
collection: goodfire-geometry
models: [Qwen3-1.7B]
fidelity: recomputed
evidence: causal
order: 630
summary: To change a model's answer from Monday to Thursday, walk along the curved weekday manifold — the outputs pass naturally through Tuesday and Wednesday.
viz: manifold-steering
data: manifold-steering.json
locations:
  - { model: qwen3-1.7b, part: resid, layers: [12, 20], label: periodic weekday spline used for along-manifold vs linear steering }
  - { model: llama-3.1-8b, part: resid, layer: 28, label: weekday / month / letter / age manifolds (Wurgaft et al.) }
tags: [steering, manifolds, splines, weekdays]
related: [calendar-circles, world-model, geometric-calculator]
citations:
  - id: wurgaft2026
    title: Manifold Steering Reveals the Shared Geometry of Neural Network Representation and Behavior
    authors: Wurgaft, Rager, Kowal, Shyam, Feucht, Bhalla, et al. (Goodfire)
    year: 2026
    url: https://arxiv.org/abs/2605.05115
  - id: ms-blog
    title: Steering Along Manifolds (Neural Geometry series)
    authors: Goodfire
    year: 2026
    url: https://www.goodfire.ai/research/manifold-steering
  - id: spokes2026
    title: Spokes, not a Ring
    authors: Gupta, Fruengel, Heimersheim
    year: 2026
    url: https://arxiv.org/abs/2610.37796
replications:
  - who: This atlas
    url: https://github.com/BurnyCoder/circuits-and-manifolds-in-llms-atlas/blob/main/src/atlas/entries/manifold_steering.py
    model: Qwen3-1.7B
    outcome: ours
    note: Periodic spline through the 7 answer centroids (64-D PCA), along-manifold vs chord paths for all day pairs 2–3 apart, behaviour-manifold energy.
  - who: Gupta, Fruengel & Heimersheim — "Spokes, not a Ring"
    url: https://arxiv.org/abs/2610.37796
    model: Llama-3.1-8B, Mistral-7B, GPT-2 XL
    outcome: partial
    note: At early layers steering along the circle is step-like (days as spokes); smooth, ring-like behaviour mainly appears late.
---

## In plain words

If a concept lives on a curve inside the model, then the straight line between two of its points cuts through empty
space. Wurgaft et al. showed this matters for steering: pushing Llama's "Monday" representation toward "Thursday" along
the curved weekday manifold makes the output glide through Tuesday and Wednesday, while the straight line produces
mixtures the model would never naturally output. We test the same idea in Qwen3-1.7B.

## How it works (technical)

The activation manifold is a periodic cubic spline through the 7 answer centroids in a 64-D PCA of the last-token
activations; the behaviour manifold is the matching spline through the centroids' output distributions in Hellinger
(√p) coordinates. Energy $E_{BC}$ is the mean Bhattacharyya distance from a path's outputs to the behaviour manifold. In
Llama-3.1-8B: $E_{BC}$ 0.34 along the manifold vs 0.93 for linear steering (weekdays).

**What we measured** (Qwen3-1.7B, best layer {{layer}}, day-arithmetic accuracy {{accuracy|pct}}, {{n_paths}} paths):

- $E_{BC}$ = {{E_manifold}} along the manifold vs {{E_linear}} along the straight line.
- {{via_manifold|pct}} of manifold path points predict one of the intermediate days vs {{via_linear|pct}} for the
  straight line.

## Limitations

- Smaller model with modest day-arithmetic accuracy; spline through 7 centroids only; we do not fit the pullback metric.
