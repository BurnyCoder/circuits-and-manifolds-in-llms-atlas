---
slug: attribution-graphs
title: Attribution graphs (the "biology" of an LLM, on Gemma-2-2B)
category: attribution
collection: anthropic-tc
models: [Gemma-2-2B]
fidelity: replicated
evidence: mixed
order: 200
summary: Feature-level causal graphs of single predictions — multi-hop facts, shared multilingual concepts, rhyme planning — on an open model. Our pre-stated re-check finds multilingual middle-layer sharing well above unrelated graphs, but the two-hop path is not specific to the intermediate concept.
viz: attribution-graph
data: attribution-graphs.json
locations:
  - { model: claude-3.5-haiku, part: band, depth: [0, 100], label: attribution graphs over a cross-layer transcoder replacement model (Ameisen et al.; Lindsey et al.) }
  - { model: qwen3-1.7b, part: mlp, layer: 14, label: transcoder features replace MLPs (circuit-tracer supports Qwen3) }
tags: [circuit tracing, transcoders, multi-hop reasoning, multilingual, planning]
related: [biology-claude, ioi, greater-than]
citations:
  - id: ameisen2025
    title: "Circuit Tracing: Revealing Computational Graphs in Language Models"
    authors: Ameisen, Lindsey, Pearce, Gurnee, Turner, Chen, Citro, et al.
    year: 2025
    venue: Transformer Circuits Thread
    url: https://transformer-circuits.pub/2025/attribution-graphs/methods.html
  - id: lindsey2025
    title: On the Biology of a Large Language Model
    authors: Lindsey, Gurnee, Ameisen, Chen, Pearce, Turner, Citro, et al.
    year: 2025
    venue: Transformer Circuits Thread
    url: https://transformer-circuits.pub/2025/attribution-graphs/biology.html
  - id: circuit-tracer
    title: circuit-tracer (open-source library, MIT)
    authors: Hanna, Piotrowski, et al.
    year: 2025
    url: https://github.com/decoderesearch/circuit-tracer
  - id: np-landscape
    title: "The Circuits Research Landscape: Results and Perspectives"
    authors: Lindsey, Ameisen, Nanda, Shabalin, Piotrowski, McGrath, Hanna, Lewis, et al.
    year: 2025
    url: https://www.neuronpedia.org/graph/info
  - id: anthropic-open
    title: Open-sourcing circuit tracing tools
    authors: Anthropic
    year: 2025
    url: https://www.anthropic.com/research/open-source-circuit-tracing
replications:
  - who: This atlas
    url: https://github.com/BurnyCoder/circuits-and-manifolds-from-mechanistic-interpretability-in-llms-and-other-models-atlas/blob/main/src/atlas/entries/attribution_graphs.py
    model: Gemma-2-2B (public Neuronpedia graphs, read-only)
    outcome: ours
    note: Re-renders nine featured graphs at supernode level and checks the two-hop path and multilingual-overlap claims quantitatively, with a label-permutation null and an unrelated-graph control in pre-stated confirmatory tests.
  - who: Neuronpedia / circuit-tracer authors — featured Gemma-2-2B graphs
    url: https://www.neuronpedia.org/graph/info
    model: Gemma-2-2B + GemmaScope transcoders
    outcome: partial
    note: Two-hop recall (with a Dallas → Austin shortcut) and language-agnostic multilingual circuits replicate on Gemma-2-2B. Rhyme features are found but match the rhyme imperfectly, and planning was not tested. Addition graphs show lookup-table features and do not recover the helix (the authors suspect transcoders).
  - who: Jacopin — planning on open models
    url: https://arxiv.org/abs/2609.18440
    model: four open models (0.6–2.6B) × six transcoders
    outcome: partial
    note: The planning "shape" replicates but the plan sits at the final prompt token, not the newline; steering at the newline barely changes the rhyme.
---

## In plain words

An attribution graph is a wiring diagram of *one* prediction. Ask "Fact: the capital of the state containing Dallas
is" and the graph shows which internal concepts fire on which words and how they pass evidence forward until the model
says "Austin". Anthropic used this to study Claude 3.5 Haiku and found readable intermediate steps — "Dallas" lights up
"Texas", "capital" lights up "say a capital", and the two combine — alongside a direct Dallas → Austin shortcut. With
the open-source circuit-tracer library the same analysis runs on open models; here we explore public graphs of
Gemma-2-2B.

## How it works (technical)

The MLPs are replaced by transcoders (sparse, interpretable features that read the residual stream and write MLP
outputs); attention patterns and normalizations are frozen for the prompt, and error nodes absorb what the transcoder
misses, so the replacement model reproduces the original output exactly. Edges are direct effects
(activation × virtual weight); the graph is pruned to the nodes that carry most of the influence on the output and
features are grouped into human-labelled supernodes. Hypotheses read off a graph are then checked with interventions
(inhibiting or swapping features) on the real model.

**What we measured** ({{n_graphs}} public Gemma-2-2B graphs, GemmaScope transcoders):

- Two-hop recall: in {{two_hop_found}} of {{two_hop_total}} capital-city graphs (Dallas, Oakland, Shanghai) there is a
  positive path from the city token through an intermediate state/country feature to the answer, next to a direct
  shortcut edge (strengths in tab 2).
- Multilingual: the English, French and Chinese antonym graphs share {{overlap_middle|pct}} of their middle-layer
  features (Jaccard), versus {{overlap_early|pct}} in early and {{overlap_late|pct}} in late layers — the language-agnostic
  middle that Lindsey et al. describe.
- Error nodes supply a median {{error_share_median|pct}} of the direct input to the output logits — the part of the
  computation the replacement model does not explain.

**Confirmatory tests** (pre-stated 2026-11-28 after the methods review flagged the two rows above as post-hoc):

- *Is the two-hop path specific to the intermediate?* No. When the labels of all feature groups are randomly permuted,
  a randomly labelled group carries a path at least as strong as the "intermediate" one often enough that the
  permutation p-values are {{label_null_p}}. The path is specific (p < 0.05) in {{two_hop_specific}} of 3 graphs.
  Paths through wrong intermediates were found in {{wrong_paths_found}} of {{wrong_paths_total}} checks, but that
  mostly means those labels do not occur in these graphs. The path check shows that strong positive paths run from
  the city to the answer; it does not show that they run through the state or country concept in particular.
- *Is the middle-layer overlap specific to the multilingual graphs?* Yes. Three unrelated graphs (Dallas, addition,
  Michael Jordan) share {{overlap_ctrl_middle|pct}} of their middle-layer features, against {{overlap_middle|pct}} for
  the three antonym graphs. One graph per language, so this is still n = 1 per condition.

## Limitations

- These graphs were generated by Neuronpedia/circuit-tracer authors; we re-render and re-analyse them but run no new
  interventions. Two-hop "intermediate concept" matching uses Neuronpedia's automatic feature labels.
- Attribution graphs explain one prompt at a time, through a replacement model; QK (attention-pattern) mechanisms are
  not explained.
- See the separate "Biology of an LLM (Claude 3.5 Haiku)" entry for the original case studies and their intervention
  results.
