---
slug: jlens
title: The Jacobian lens and the global workspace (open model)
category: lens
collection: anthropic-tc
models: [Qwen3-1.7B]
fidelity: replicated
evidence: mixed
order: 500
summary: A lens that reads what each layer "is about to say" exposes unspoken intermediate steps in Claude, and swapping its coordinates swaps the concept. In Qwen3-1.7B the J-lens beats the logit lens on most of 24 fresh prompts, narrowly missing significance, and the swaps fail.
viz: jlens
data: jlens.json
locations:
  - { model: qwen3-1.7b, part: resid, layers: [8, 20], label: J-lens readouts of intermediate concepts; coordinate swaps tested }
  - { model: claude-sonnet-4.5, part: band, depth: [38, 92], label: verbalizable global workspace (J-space) band (Gurnee et al. 2026) }
tags: [logit lens, Jacobian lens, global workspace, latent reasoning]
related: [attribution-graphs, global-workspace-claude]
quote:
  text: These properties make it a practical window into a model's unspoken thinking.
  source: gurnee2026ws
  where: Abstract
citations:
  - id: gurnee2026ws
    title: Verbalizable Representations Form a Global Workspace in Language Models
    authors: Gurnee, Sofroniew, Pearce, Piotrowski, … Lindsey
    year: 2026
    venue: Transformer Circuits Thread (arXiv 2607.15495)
    url: https://transformer-circuits.pub/2026/workspace/index.html
  - id: jlens-code
    title: jacobian-lens (reference implementation, Apache-2.0)
    authors: Anthropic
    year: 2026
    url: https://github.com/anthropics/jacobian-lens
  - id: np-lenses
    title: Pre-fitted Jacobian lenses (Hugging Face neuronpedia/jacobian-lens)
    authors: Neuronpedia
    year: 2026
    url: https://huggingface.co/neuronpedia/jacobian-lens
  - id: nostalgebraist2020
    title: interpreting GPT — the logit lens
    authors: nostalgebraist
    year: 2020
    venue: LessWrong
    url: https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens
replications:
  - who: This atlas
    url: https://github.com/BurnyCoder/circuits-and-manifolds-in-llms-atlas/blob/main/src/atlas/entries/jlens.py
    model: Qwen3-1.7B + Neuronpedia's pre-fitted lens
    outcome: ours
    note: Slice grids vs the logit lens, intermediate-token ranks, and coordinate swaps with a random-direction control.
  - who: Viswanath — "Models are blind outside the J-space. NLAs aren't."
    url: https://www.lesswrong.com/posts/LhDJdccLszLEAqgZ9/models-are-blind-outside-the-j-space-nlas-aren-t
    model: Llama-3.3-70B
    outcome: replicated
    note: The J-space part of concept vectors holds ~6.1% of their energy (paper 6–7%); injections outside J-space are never verbalized.
  - who: avalur/JLens
    url: https://github.com/avalur/JLens
    model: Qwen2.5-1.5B/7B, Gemma-2-2B/9B
    outcome: partial
    note: Early detection and the France → China broadcast reproduce; ablation selectivity does not reproduce up to 9B.
  - who: Bandularatne — failed GPT-2 replication
    url: https://www.lesswrong.com/posts/tgn3pD2gLpZvepkWk/j-lens-a-failed-replication-on-gpt-2
    model: GPT-2 small / medium
    outcome: failed
    note: The J-lens loses to the logit lens at every layer (0/11 on GPT-2 small, 1/23 on medium), and the gap widens with scale; a commenter suggests weight tying.
---

## In plain words

The "logit lens" asks every layer: if the model had to answer right now, what would it say? Anthropic's Jacobian lens
does this more faithfully by accounting for how later layers transform each layer's state. Applied to Claude it reveals
a middle "workspace" where the model holds concepts it never writes down — for "(4+17)·2+7" it holds 21, then 42, before
saying 49 — and editing those concepts changes the answer. We use a lens fitted for the open Qwen3-1.7B.

## How it works (technical)

$J_\ell = \mathbb{E}\big[\partial h_{\text{late}, t'} / \partial h_{\ell, t}\big]$, averaged over prompts and positions; the
lens reads $\text{softmax}(W_U\, \text{norm}(J_\ell h_\ell))$ (the logit lens is $J_\ell = I$). In Claude Sonnet 4.5 the readable
band spans roughly 38–92% of depth with ~25 concepts at a time; swapping the J-lens coordinates of two concepts (France →
China) changes capital, language and continent answers together (76/192 top-1 across 16 functions; 101/192 at double
strength). A separate "think of a {category}" swap succeeds in the top 5 for 59% of trials with the J-space component of
the concept vector vs 5% with the rest.

**What we measured** (Qwen3-1.7B, pre-fitted lens over {{n_layers_lens}} layers):

- For {{n_inter}} unspoken intermediates (Texas for Dallas; France for Paris and for the Eiffel Tower) the best rank at the last position has median
  {{inter_median_rank_j}} with the J-lens vs {{inter_median_rank_logit}} with the logit lens; {{inter_top10_j}} vs
  {{inter_top10_logit}} reach the top 10. This row is post-hoc: the arithmetic prompt (21, 42) was removed after its
  ranks, which favoured the logit lens, had been seen.
- **Confirmatory test** (pre-stated before running, 24 fresh two-hop prompts with one-token intermediates): the J-lens
  ranks the intermediate better on {{confirm_wins_j|int}} prompts, the logit lens on {{confirm_wins_logit|int}}, and
  they tie on the rest; two-sided sign test p = {{confirm_sign_p}}, just above the pre-stated 0.05, so the verdict is
  partial. Median best rank {{confirm_median_rank_j}} vs {{confirm_median_rank_logit}}; top 10 for
  {{confirm_top10_j|int}} vs {{confirm_top10_logit|int}} of {{confirm_n|int}}. See tab 4 of the visualization.
- Swapping the J-lens coordinates of two countries at the country token (layer {{swap_layer}}) puts the other country's
  answer in the top 5 for {{swap_top5|pct}} of {{swap_n}} prompts, vs {{swap_random|pct}} for a random direction.

## Limitations

- Small model, few prompts, single-token concepts only.
- The lens was fitted by Neuronpedia on WikiText; we did not refit it. The paper's selectivity, occupancy and auditing
  results are not tested here.
