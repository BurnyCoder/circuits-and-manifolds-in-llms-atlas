---
slug: global-workspace-claude
title: A global workspace in Claude (J-space)
category: lens
collection: anthropic-tc
models: [Claude Sonnet 4.5, Claude Haiku 4.5, Claude Opus 4.5]
fidelity: redrawn
evidence: causal
order: 505
summary: A small, verbalizable set of concepts in Claude's middle layers works like a global workspace — reportable, steerable and used for silent multi-step reasoning.
viz: workspace-claude
locations:
  - { model: claude-sonnet-4.5, part: band, depth: [38, 92], label: J-space workspace band (~25 concepts at a time) }
tags: [Jacobian lens, global workspace, introspection, latent reasoning, auditing]
related: [jlens, biology-claude, emotions-claude]
citations:
  - id: gurnee2026ws
    title: Verbalizable Representations Form a Global Workspace in Language Models
    authors: Gurnee, Sofroniew, Pearce, Piotrowski, Kauvar, Chen, Soligo, Bogdan, Ong, Wang, Thompson, Abrahams, Kantamneni, Ameisen, Batson, Lindsey
    year: 2026
    venue: Transformer Circuits Thread (arXiv 2607.15495)
    url: https://transformer-circuits.pub/2026/workspace/index.html
  - id: ws-blog
    title: A global workspace in language models (blog post)
    authors: Anthropic
    year: 2026
    url: https://www.anthropic.com/research/global-workspace
  - id: jpp
    title: "Circuits Updates — October 2026: the J++ lens"
    authors: Ayonrinde, Lindsey
    year: 2026
    url: https://transformer-circuits.pub/2026/october-update/index.html
  - id: jpp-repo
    title: "jpp_lens (code and recall@10 numbers for the J++ lens)"
    authors: Ayonrinde, Lindsey (safety-research)
    year: 2026
    url: https://github.com/safety-research/jpp_lens
replications:
  - who: This atlas (Qwen3-1.7B)
    url: https://github.com/BurnyCoder/circuits-and-manifolds-in-llms-atlas/blob/main/src/atlas/entries/jlens.py
    model: Qwen3-1.7B
    outcome: ours
    note: J-lens slice grids, intermediate-token ranks and coordinate swaps on an open model (see the open entry).
  - who: Viswanath
    url: https://www.lesswrong.com/posts/LhDJdccLszLEAqgZ9/models-are-blind-outside-the-j-space-nlas-aren-t
    model: Llama-3.3-70B
    outcome: replicated
    note: J-space share of concept energy ≈ 6.1% (paper 6–7%).
  - who: avalur/JLens
    url: https://github.com/avalur/JLens
    model: Qwen2.5, Gemma-2 (≤ 9B)
    outcome: partial
    note: Broadcast reproduces; selectivity does not reproduce up to 9B.
---

## In plain words

Psychologists describe a "global workspace": a small stage of things you are currently aware of and can talk about,
while lots of processing happens backstage. Anthropic found something analogous in Claude. A lens that reads what each
layer "would say" exposes a band of middle layers holding a couple of dozen concepts at once — the ones the model can
report, hold on purpose, reason with silently (21, then 42, before answering 49), and pass along. Swapping one concept
there (France → China) changes its answers about capital, language and continent together (one example; 76/192 top-1
across categories); wiping it out
destroys multi-step reasoning while routine tasks survive.

## How it works (technical)

The Jacobian lens reads layer ℓ through the prompt-averaged Jacobian $J_\ell$ of a late layer w.r.t. layer ℓ; J-space is a
sparse non-negative combination of up to ~25 J-lens vectors. Published results: workspace band ≈ 38–92% of depth; concept
swaps succeed in the top 5 for 59% (J-space component) vs 5% (non-J component); multi-hop coordinate swaps 54–70% across
Haiku/Sonnet/Opus 4.5; ablating broadcast heads drops injected-thought reports from 0.54 to 0.09; ablating
evaluation-awareness directions cuts verbalized eval awareness from 71% to 3%. The October 2026 J++ lens raises recall@10
to 55.2% vs 35.7% for the J-lens on Qwen3.6-27B (numbers from the jpp_lens repository README).

## Limitations

- Closed models; single-token concepts; the authors take no position on phenomenal consciousness.
