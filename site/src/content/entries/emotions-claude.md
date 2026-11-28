---
slug: emotions-claude
title: Emotion concepts and their function (Claude Sonnet 4.5)
category: steering
collection: anthropic-tc
models: [Claude Sonnet 4.5]
fidelity: redrawn
evidence: causal
order: 425
summary: 171 emotion vectors in Claude organise like human emotions — and steering "desperate" or "calm" moves blackmail and reward-hacking rates up and down.
viz: emotions-claude
locations:
  - { model: claude-sonnet-4.5, part: band, depth: [60, 72], label: emotion vectors at ~2/3 depth }
tags: [emotions, steering, misalignment, functional emotions]
related: [emotions, persona-vectors-claude, refusal-direction]
citations:
  - id: sofroniew2026
    title: Emotion Concepts and their Function in a Large Language Model
    authors: Sofroniew, Kauvar, Saunders, Chen, Henighan, … Olah, Lindsey
    year: 2026
    venue: Transformer Circuits Thread (arXiv 2604.07729)
    url: https://transformer-circuits.pub/2026/emotions/index.html
  - id: blog
    title: Emotion concepts in a large language model (blog post)
    authors: Anthropic
    year: 2026
    url: https://www.anthropic.com/research/emotion-concepts-function
  - id: viewer
    title: Transcript visualizations (on-policy emotion probe viewer)
    authors: Anthropic
    year: 2026
    url: https://transformer-circuits.pub/2026/emotions/onpolicy/index.html?t=Blackmail
replications:
  - who: This atlas (Qwen3-1.7B)
    url: https://github.com/BurnyCoder/circuits-and-manifolds-in-llms-atlas/blob/main/src/atlas/entries/emotions.py
    model: Qwen3-1.7B
    outcome: ours
    note: Valence/arousal geometry, dose sweep and preference steering on an open model (see the open entry).
  - who: Hong — Llama-3.1-8B-Instruct
    url: https://jyhong.gitlab.io/post/emotion-vector-part1/
    model: Llama-3.1-8B-Instruct
    outcome: partial
    note: 10/11 checks pass; implicit-emotion test fails; much smaller effects.
  - who: Jain et al.
    url: https://lesswrong.com/posts/PYcqf9DBMGjTyJSjg/model-s-emotions-noticeably-change-when-they-take-risky-1
    model: Gemma-3, GPT-OSS-120B, GLM
    outcome: partial
    note: Steering toward "calm" often increased reward hacking — opposite to the paper.
---

## In plain words

Claude wrote ~205,000 short stories about characters feeling one of 171 emotions; averaging its activations over each
emotion's stories gives an "emotion vector". The vectors arrange like human emotions (pleasant–unpleasant first,
calm–excited second), they react to situations (a rising medicine dose raises "afraid"), and they drive behaviour: in an
agentic test where the model could blackmail someone, adding a little "desperation" raised the blackmail rate from 22% to
72%, while adding "calm" brought it to 0%. The authors stress these are functional representations, not evidence of
feelings.

## How it works (technical)

Vector = mean residual activation over a story's tokens (from token 50) minus the mean over all emotions, with neutral-text
PCs removed, at about two-thirds depth. PC1 (26%) ↔ human valence (r = 0.81), PC2 (15%) ↔ arousal (r = 0.66). Steering at
0.5 × residual norm on an activity's tokens changes its Elo (blissful +212, hostile −303; r = 0.85 with probe correlations).
Blackmail: unsteered 22%; desperate +0.05 → 72%; desperate −0.05 or calm +0.05 → 0%; calm −0.05 → 66%. Reward hacking:
desperate −0.1 → ~5%, +0.1 → ~70%.

## Limitations

- Closed model; contrived agentic prompts; vectors assume linear emotion directions and are trained on synthetic,
  stereotyped stories (the paper's own limitations).
