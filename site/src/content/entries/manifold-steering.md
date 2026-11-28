---
slug: manifold-steering
title: Steering along manifolds
category: manifold
collection: goodfire-geometry
models: [Qwen3-1.7B]
fidelity: recomputed
evidence: causal
order: 630
summary: To change Llama's answer from Monday to Thursday, walk along the curved weekday manifold — the outputs pass through Tuesday and Wednesday. In Qwen3-1.7B the path visits the intermediate days, as built, but a smooth path is no better than straight segments through the same day centroids.
viz: manifold-steering
data: manifold-steering.json
locations:
  - { model: qwen3-1.7b, part: resid, layers: [22, 26], label: periodic weekday spline used for along-manifold vs linear steering }
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
    url: https://github.com/BurnyCoder/circuits-and-manifolds-from-mechanistic-interpretability-in-llms-and-other-models-atlas/blob/main/src/atlas/entries/manifold_steering.py
    model: Qwen3-1.7B
    outcome: ours
    note: Periodic spline through the 7 answer centroids (in their 6-D span; layers 22–26), along-manifold vs chord paths for all day pairs 2–3 apart, behaviour-manifold energy. Both original rows are post-hoc (layers chosen after failing runs at 12–20); a pre-stated confirmatory test against a piecewise-linear path through the same centroids fails at all three layers.
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

In the paper the activation manifold is a periodic cubic spline through the 7 answer centroids in a 64-D PCA of the
last-token activations (we use the centroids' 6-D span, because ~60 prompts are too few for a 64-D PCA); the behaviour manifold is the matching spline through the centroids' output distributions in Hellinger
(√p) coordinates. Energy $E_{BC}$ is the mean Bhattacharyya distance from a path's outputs to the behaviour manifold. In
Llama-3.1-8B: $E_{BC}$ 0.34 along the manifold vs 0.93 for linear steering (weekdays).

**What we measured** (Qwen3-1.7B, best layer {{layer}}, day-arithmetic accuracy {{accuracy|pct}}, {{n_paths}} paths):

- $E_{BC}$ = {{E_manifold}} along the manifold vs {{E_linear}} along the straight line.
- {{via_manifold|pct}} of manifold path points predict one of the intermediate days vs {{via_linear|pct}} for the
  straight line. This is close to guaranteed by construction: the spline passes exactly through the intermediate
  centroids, which make the model say that day almost every time at these layers, while a chord between two vertices
  never passes a third. A piecewise-linear path through the same centroids visits them about as often (verdict
  table, row 2).
- Both rows are post-hoc: the layers (22–26) were chosen after full runs at layers 12–20 had failed. At layers 12–20,
  writing a day's centroid makes the model say that day in almost no cases (endpoint success by layer: {{endpoint_by_layer}}),
  so the answer is not yet formed there.
- Confirmatory test (pre-stated 2026-11-28; layers 22, 24 and 26 all used, 4 source prompts per day): does the smooth
  manifold path stay closer to natural outputs than a piecewise-linear path through the *same* centroids? No:
  $E_{BC}$(spline) − $E_{BC}$(piecewise-linear) = {{spline_vs_polyline}} (mean over paths, 95% CI over source prompts),
  positive at every layer. In Qwen3-1.7B the curved geometry between the day centroids does not help; what moves the
  answer through the intermediate days is passing through their centroids.

## Limitations

- Smaller model with modest day-arithmetic accuracy; spline through 7 centroids only; we do not fit the pullback metric.
