---
slug: sae-features
title: SAE features and "Golden Gate" steering (GPT-2)
category: feature
collection: anthropic-tc
models: [GPT-2 small]
fidelity: recomputed
evidence: causal
order: 210
summary: A sparse autoencoder splits GPT-2's activations into readable features, and turning one up steers the text. The closest GPT-2 analogue of Golden Gate Claude is a generic bridge feature — it steers toward bridges, not toward the Golden Gate.
viz: sae-features
data: sae-features.json
locations:
  - { model: gpt2-small, part: resid, layer: 7, label: "SAE features (gpt2-small-res-jb, residual entering block 8): Golden Gate, Python, French, DNA, Harry Potter" }
  - { model: claude-sonnet-4.5, part: band, depth: [45, 55], label: "Scaling Monosemanticity used Claude 3 Sonnet's middle layer (shown on the Claude diagram)" }
tags: [sparse autoencoders, monosemanticity, feature steering, Golden Gate]
related: [scaling-monosemanticity-claude, toy-superposition, sae-manifolds]
citations:
  - id: templeton2024
    title: "Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet"
    authors: Templeton, Conerly, Marcus, Lindsey, Bricken, Chen, Pearce, Citro, et al.
    year: 2024
    venue: Transformer Circuits Thread
    url: https://transformer-circuits.pub/2024/scaling-monosemanticity/
  - id: bricken2023
    title: "Towards Monosemanticity: Decomposing Language Models With Dictionary Learning"
    authors: Bricken, Templeton, et al.
    year: 2023
    venue: Transformer Circuits Thread
    url: https://transformer-circuits.pub/2023/monosemantic-features
  - id: saelens
    title: SAELens and the gpt2-small-res-jb SAEs
    authors: Bloom, Tigges, Duong, Chanin, et al.
    year: 2024
    url: https://decoderesearch.github.io/SAELens/latest/pretrained_saes/gpt2-small/
  - id: neuronpedia
    title: Neuronpedia (feature dashboards)
    authors: Lin, Bloom, et al.
    year: 2024
    url: https://www.neuronpedia.org/gpt2-small
replications:
  - who: This atlas
    url: https://github.com/BurnyCoder/circuits-and-manifolds-in-llms-atlas/blob/main/src/atlas/entries/sae_features.py
    model: GPT-2 small + gpt2-small-res-jb (layer 8)
    outcome: ours
    note: Feature selection with a specificity control, top contexts, logit effects, and steering sweeps with norm-matched random directions.
  - who: Louapre — "The Eiffel Tower Llama"
    url: https://huggingface.co/spaces/dlouapre/eiffel-tower-llama
    model: Llama-3.1-8B-Instruct + 131k-latent SAE
    outcome: partial
    note: Feature clamping works in a narrow sweet spot but still loses to a simple system prompt on fluency and judged quality.
  - who: AxBench
    url: https://arxiv.org/abs/2501.17148
    model: Gemma-2 + Gemma Scope SAEs
    outcome: partial
    note: Prompting beats fine-tuning, which beats SAE-feature steering, on a concept-steering benchmark.
---

## In plain words

A sparse autoencoder (SAE) rewrites each activation as a combination of a few "features" out of tens of thousands, and
many features turn out to mean one thing: the Golden Gate Bridge, Python code, French text. Anthropic famously turned up
Claude's Golden Gate Bridge feature until the model believed it *was* the bridge ("Golden Gate Claude"). Claude's SAEs are
private, but open SAEs exist for GPT-2, so here we look for a Golden Gate feature in GPT-2 small and steer with it. The
best match is a feature for bridges in general.

## How it works (technical)

An SAE encodes $f = \text{ReLU}(W_{\text{enc}}(x - b) + b_{\text{enc}})$ and decodes $\hat x = W_{\text{dec}} f + b$, trained to
reconstruct with an L1 sparsity penalty. Feature steering adds $\alpha\,W_{\text{dec}}[f]$ to the residual stream.
Templeton et al. trained SAEs with 1M–34M features on Claude 3 Sonnet, found multilingual, multimodal and safety-relevant
features, and showed clamping changes behaviour.

**What we measured** (GPT-2 small, layer 8, {{n_concepts}} concepts):

- The bridge feature (#{{gg_feature}}, the top Neuronpedia match for "Golden Gate Bridge"; its label is about bridges in
  general): completions from neutral prompts mention a bridge, the Golden Gate or San Francisco in {{gg_base|pct}} of
  samples without steering, {{gg_steer|pct}} with α = {{alpha}}, and {{gg_random|pct}} for a random direction of the same
  norm. Only {{gg_specific|pct}} of the steered completions name the Golden Gate or San Francisco.
- {{n_lift}} of {{n_concepts}} concept features gain ≥ 30 points of on-topic completions at α = {{alpha}} (median lift
  {{median_lift|pct}}).
- Both rows are post-hoc: the feature-selection rule was changed twice after smoke tests that had logged steering
  rates. Since the review each cell has {{n_per_cell}} completions (was 12), with Wilson 95% CIs in the verdict table;
  the outcome is unchanged: the feature steers toward bridges in general, never toward the Golden Gate in particular.

## Limitations

- GPT-2 small is a weak writer; strong steering degrades fluency. Features were chosen with a simple specificity score,
  and keyword matching is a coarse judge.
