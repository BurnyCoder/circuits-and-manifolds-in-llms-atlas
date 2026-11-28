---
slug: world-model
title: The world inside a world model (mountain car)
category: manifold
collection: goodfire-geometry
models: [mountain-car world model (trained here)]
fidelity: recomputed
evidence: mixed
order: 600
summary: Goodfire report that a video-predicting network places every frame on a curve indexed by the car's position — walk along it and the car moves; cut across and it teleports. In our rerun an untrained encoder shows the same curve.
viz: world-model
data: world-model.json
locations:
  - { model: mountain-car-wm, part: stage, name: latent, label: frame embeddings lie on a 1-D curve indexed by the car's position }
  - { model: mountain-car-wm, part: stage, name: GRU, label: predicts the next frame from the latent and the action }
tags: [world models, manifolds, interpolation, reinforcement learning]
related: [manifold-steering, othello-gpt, calendar-circles]
citations:
  - id: geiger2026
    title: The World Inside Neural Networks (Neural Geometry series)
    authors: Geiger, Lubana, Fel, Merullo, Byun, Lewis, McGrath (Goodfire)
    year: 2026
    url: https://www.goodfire.com/research/the-world-inside-neural-networks
  - id: wurgaft2026
    title: Manifold Steering Reveals the Shared Geometry of Neural Network Representation and Behavior (App. B, mountain car)
    authors: Wurgaft, Rager, Kowal, et al.
    year: 2026
    url: https://arxiv.org/abs/2605.05115
  - id: gymnasium
    title: Gymnasium MountainCar-v0
    authors: Farama Foundation
    year: 2023
    url: https://gymnasium.farama.org/environments/classic_control/mountain_car/
replications:
  - who: This atlas
    url: https://github.com/BurnyCoder/circuits-and-manifolds-in-llms-atlas/blob/main/src/atlas/entries/world_model.py
    model: conv encoder + GRU world model (64×64), trained here
    outcome: ours
    note: Same architecture and data recipe at lower resolution, with an auxiliary frame decoder for visualization and an untrained-encoder control.
---

## In plain words

Goodfire's essay argues that neural networks build an inner "world" whose shape mirrors the real one. Their clearest
example is tiny: a network that watches the classic mountain-car game and predicts the next frame. The car's position is
one number, and the network ends up arranging all frames along a single curved string in its internal space, ordered
by position. Following that string from the valley to the hilltop gives sensible frames; drawing a straight line between
the two ends passes through "voids" where decoded frames are garbled or the car jumps.

## How it works (technical)

A conv encoder maps each frame to a 64-D LayerNorm latent, a GRU combines it with the action, and a decoder predicts the
next frame as a residual. The manifold is a smoothing spline through position-binned means of the latents. Isometry:
correlation between distances along the path and true position differences over 50 anchors (the paper's metric differs:
it correlates arc lengths on the activation manifold with arc lengths on the behaviour manifold, r = 0.996 over 1,225
anchor pairs, vs r = 0.06 for activation-space chords).

**What we measured** ({{n_frames}} frames from 100 rollouts):

- Linear probes read position with $R^2$ = {{r2_position}} and velocity with {{r2_velocity}} (untrained encoder: position
  {{r2_position_untrained}}).
- Distances along the spline track position differences with r = {{r_arc}}; straight-chord distances give r = {{r_chord}}.
- Control added after the methods review: the same analysis on the **untrained** encoder gives r = {{r_arc_untrained}}
  along its spline and {{r_chord_untrained}} for chords. A spline through position-ordered means is monotone in
  position almost by construction, so this isometry does not show anything learned (the control row fails).
- Mean distance from path points to the nearest real frame embedding: {{void_manifold}} along the manifold vs
  {{void_linear}} along the straight line.

## Limitations

- 64×64 frames and our own training run; the auxiliary current-frame decoder (for visualization) is our addition. The
  slant-rhyme manifold from the same essay is not reproduced (pipeline unpublished).
