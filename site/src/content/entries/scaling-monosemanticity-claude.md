---
slug: scaling-monosemanticity-claude
title: Scaling monosemanticity and Golden Gate Claude (Claude 3 Sonnet)
category: feature
collection: anthropic-tc
models: [Claude 3 Sonnet]
fidelity: redrawn
evidence: causal
order: 215
summary: Millions of interpretable features in a production model — from the Golden Gate Bridge to code vulnerabilities and deception — and clamping one changes who the model thinks it is.
viz: scaling-mono
locations:
  - { model: claude-sonnet-4.5, part: band, depth: [45, 55], label: "Claude 3 Sonnet middle-layer SAEs (1M / 4M / 34M features) — drawn on the Claude diagram" }
tags: [sparse autoencoders, feature splitting, safety features, steering]
related: [sae-features, toy-superposition, sae-manifolds]
citations:
  - id: templeton2024
    title: "Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet"
    authors: Templeton, Conerly, Marcus, Lindsey, Bricken, Chen, Pearce, Citro, et al.
    year: 2024
    venue: Transformer Circuits Thread
    url: https://transformer-circuits.pub/2024/scaling-monosemanticity/
  - id: ggc
    title: Golden Gate Claude
    authors: Anthropic
    year: 2024
    url: https://www.anthropic.com/news/golden-gate-claude
  - id: mapping
    title: Mapping the Mind of a Large Language Model
    authors: Anthropic
    year: 2024
    url: https://www.anthropic.com/research/mapping-mind-language-model
replications:
  - who: Louapre — "The Eiffel Tower Llama"
    url: https://huggingface.co/spaces/dlouapre/eiffel-tower-llama
    model: Llama-3.1-8B-Instruct
    outcome: partial
    note: Clamping an SAE feature reproduces the obsession effect in a narrow coefficient range; a system prompt still does better.
  - who: This atlas — Golden Gate GPT-2
    url: https://github.com/BurnyCoder/circuits-and-manifolds-in-llms-atlas/blob/main/src/atlas/entries/sae_features.py
    model: GPT-2 small
    outcome: ours
    note: Open version with a Golden Gate feature, top activations and steering sweeps.
---

## In plain words

Anthropic trained sparse autoencoders with up to 34 million features on Claude 3 Sonnet and found features for almost
everything: famous landmarks, people, programming errors, sycophancy, deception, dangerous topics — often responding to
the concept across languages and even images. Features also "split": one San Francisco feature in a small dictionary
becomes many finer ones in bigger dictionaries. Turning a single feature up to ten times its normal maximum made Claude
believe it was the Golden Gate Bridge ("Golden Gate Claude", a public demo in May 2024).

## How it works (technical)

Three dictionaries (1M, 4M, 34M features) on the middle-layer residual stream; dead-feature fractions ≈ 2%, 35% and 65%;
fewer than 300 features active per token; reconstruction explains at least 65% of variance; loss follows a power law in
compute. Feature steering clamps a feature's activation and lets the decoder direction flow through the rest of the
model. The arXiv copy of the paper (2605.29358) carries a CC BY 4.0 license.

## Limitations

- Closed model and weights: all numbers are from the paper and blog posts; the open analogue is the GPT-2 entry.
