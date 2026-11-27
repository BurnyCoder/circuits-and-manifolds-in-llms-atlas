---
slug: biology-claude
title: On the Biology of a Large Language Model (Claude 3.5 Haiku)
category: attribution
collection: anthropic-tc
models: [Claude 3.5 Haiku]
fidelity: redrawn
evidence: causal
order: 205
summary: Anthropic traced how Claude plans rhymes, chains facts, adds numbers, recognises entities and gets jailbroken — interactive redraws of eight case studies.
viz: biology-claude
locations:
  - { model: claude-3.5-haiku, part: band, depth: [0, 100], label: case-study attribution graphs span the whole model (cross-layer transcoder replacement) }
tags: [attribution graphs, planning, multi-hop reasoning, hallucination, jailbreaks, chain of thought]
related: [attribution-graphs, linebreaks-claude, jlens]
citations:
  - id: lindsey2025
    title: On the Biology of a Large Language Model
    authors: Lindsey, Gurnee, Ameisen, Chen, Pearce, Turner, Citro, et al.
    year: 2025
    venue: Transformer Circuits Thread
    url: https://transformer-circuits.pub/2025/attribution-graphs/biology.html
  - id: ameisen2025
    title: "Circuit Tracing: Revealing Computational Graphs in Language Models"
    authors: Ameisen, Lindsey, Pearce, Gurnee, Turner, Chen, Citro, et al.
    year: 2025
    venue: Transformer Circuits Thread
    url: https://transformer-circuits.pub/2025/attribution-graphs/methods.html
replications:
  - who: Neuronpedia / circuit-tracer (Gemma-2-2B)
    url: https://www.neuronpedia.org/graph/info
    model: Gemma-2-2B
    outcome: partial
    note: Two-hop recall and multilingual circuits replicate; rhyme planning partial; addition shows lookup tables instead.
  - who: Jacopin — "Planning or Improvisation?"
    url: https://arxiv.org/abs/2609.18440
    model: four open models (0.6–2.6B)
    outcome: partial
    note: The planning figure's shape replicates but not its mechanism (plan at the final prompt token, not the newline).
  - who: Ferrando et al. — "Do I Know This Entity?"
    url: https://arxiv.org/abs/2411.14257
    model: Gemma-2 + Gemma Scope SAEs
    outcome: replicated
    note: Known- and unknown-entity latents causally steer refusing vs hallucinating, mirroring the "can't answer" default circuit.
---

## In plain words

Anthropic built a "microscope" for Claude 3.5 Haiku: replace its MLPs with interpretable features, then trace, for one
prompt at a time, which features cause which until the answer comes out. They used it like biologists studying an
organism and found surprisingly concrete behaviours — the model picks a rhyme word before writing the line, chains
"Dallas → Texas → Austin" in its head, adds numbers with two parallel paths (a rough estimate and an exact last digit),
refuses by default unless it recognises a name, and can be tricked by spelling a dangerous word letter by letter.

## How it works (technical)

Attribution graphs are computed on a cross-layer transcoder replacement model (up to 30M features) with frozen attention
and error nodes; supernodes group related features, and each hypothesis is tested by inhibiting, injecting or swapping
features in the real model. The visualization redraws each case's core diagram with paraphrased labels and lists the
published intervention outcomes — e.g. swapping the "Texas" features for California/Georgia/British Columbia/China/
Byzantine Empire changes the answer to Sacramento/Atlanta/Victoria/Beijing/Constantinople; injecting a planned word makes
lines end with it about 70% of the time over 25 poems; inhibiting reward-model-bias features cuts meta-rhymes from ~48% to
~12%.

## Limitations

- Closed model: every number here is from the paper; open-model graphs are in the Gemma-2-2B entry.
- The method explains one prompt at a time through a replacement model; attention-pattern (QK) formation and error-node
  "dark matter" remain unexplained (paper §Limitations).
