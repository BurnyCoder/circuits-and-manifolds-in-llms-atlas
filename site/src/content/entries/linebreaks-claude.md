---
slug: linebreaks-claude
title: Manipulating manifolds to break lines (Claude 3.5 Haiku)
category: manifold
collection: anthropic-tc
models: [Claude 3.5 Haiku]
fidelity: redrawn
evidence: causal
order: 125
summary: Claude counts characters on a rippled curve, and attention heads rotate that curve onto the line width to sense when a line is full — live simulations of the mechanism.
viz: linebreaks-claude
locations:
  - { model: claude-3.5-haiku, part: band, depth: [0, 12], label: layer-0/1 heads build the character-count manifold (~6-D, rippled) }
  - { model: claude-3.5-haiku, part: band, depth: [10, 30], label: boundary heads rotate count onto width (QK); characters-remaining subspace }
tags: [counting, manifolds, attention, QK circuits, visual illusions]
related: [linebreak-manifold, attribution-graphs, biology-claude]
citations:
  - id: gurnee2025
    title: "When Models Manipulate Manifolds: The Geometry of a Counting Task"
    authors: Gurnee, Ameisen, Kauvar, Tarng, Pearce, Olah, Batson
    year: 2025
    venue: Transformer Circuits Thread (arXiv 2601.04480)
    url: https://transformer-circuits.pub/2025/linebreaks/index.html
  - id: modell2025
    title: The Origins of Representation Manifolds in Large Language Models
    authors: Modell, Rubin-Delanchy, Whiteley
    year: 2025
    url: https://arxiv.org/abs/2505.18235
replications:
  - who: Sinii & Balagansky (T-Tech) — counting manifold in open LLMs
    url: https://t-tech-manifolds.hf.space/
    model: GPT-2, Pythia, Gemma, Qwen3, Llama
    outcome: partial
    note: Decodable count and a helix with ~155-character period replicate; the very-early-layer peak does not; QK twisting untested.
  - who: This atlas (Qwen3-1.7B)
    url: https://github.com/BurnyCoder/circuits-and-manifolds-in-llms-atlas/blob/main/src/atlas/entries/linebreak_manifold.py
    model: Qwen3-1.7B
    outcome: ours
    note: See the open-model entry for our probe, manifold and ringing measurements.
---

## In plain words

To wrap text at a fixed width, Claude must know how far along the line it is and whether the next word will fit. The
paper shows how: early attention heads add up the lengths of the tokens since the last newline, producing a character
count stored on a curve that twists through about six dimensions, read by ten or so "place-cell" features. Then
"boundary heads" compare this count with the line width — their attention mechanism literally rotates one curve onto the
other — so the model sees how many characters remain, and a final step compares that with the length of the next word.
Inserting odd tokens like "@@" can fool the counting heads, a kind of visual illusion for the model.

## How it works (technical)

The layer-2 residual means for counts 1–150 put 95% of their variance in 6 principal components; a linear probe after
layer 1 reaches $R^2$ = 0.985. Cosine-similarity matrices "ring" — a diagonal band plus echoes — which the authors explain
with an optimal-packing argument and a particle simulation (points on an $n$-sphere attracting neighbours and repelling
the rest: the second tab runs it live). A boundary head's QK matrix maps count $i$ onto width $i + \varepsilon$: count–width
probe similarity is ≈ 0.25 in the residual stream but ≈ 1 after QK; about three heads with different offsets tile
"characters remaining" (2 PCs hold 92%), and characters-remaining vs next-word length are separated by a hyperplane (AUC 0.91).

## Limitations

- Closed model: the simulations illustrate the published mechanism; numbers are the paper's. See the open-model entry for
  measured geometry in Qwen3-1.7B.
