---
slug: actadd
title: Activation addition (steering vectors)
category: steering
collection: steering
models: [GPT-2-XL]
status: pending
fidelity: recomputed
evidence: causal
order: 410
summary: Subtract the activations of "I do not talk about weddings" from "I talk about weddings constantly", add the difference while the model writes — and it talks about weddings.
viz: actadd
data: actadd.json
tags: [steering vectors, activation engineering, contrast pairs]
related: [refusal-direction, sae-features, emotions]
citations:
  - id: turner2023
    title: Steering Language Models With Activation Engineering
    authors: Turner, Thiergart, Leech, Udell, Vazquez, Mini, MacDiarmid
    year: 2023
    venue: arXiv (CC BY 4.0)
    url: https://arxiv.org/abs/2308.10248
  - id: zou2023
    title: "Representation Engineering: A Top-Down Approach to AI Transparency"
    authors: Zou, Phan, Chen, Campbell, Guo, et al.
    year: 2023
    url: https://arxiv.org/abs/2310.01405
replications:
  - who: This atlas
    url: https://github.com/BurnyCoder/circuits-and-manifolds-in-llms-atlas/blob/main/src/atlas/entries/actadd.py
    model: GPT-2-XL
    outcome: ours
    note: Coefficient sweeps for two vectors, keyword and sentiment-classifier scoring, random-vector control; all completions shown.
---

## In plain words

You can change what a language model talks about without training it, by nudging its internal activations. Take two
short prompts that differ in one idea ("Love" vs "Hate", or "I talk about weddings constantly" vs "I do not…"), record the
difference in the model's activations, and add a multiple of that difference while the model generates. GPT-2-XL then
writes happier text, or steers every conversation toward weddings.

## How it works (technical)

ActAdd computes $v = h_\ell(p_+) - h_\ell(p_-)$ for a contrast pair at layer $\ell$ (padded to equal length) and adds
$c\,v$ to the first positions of the user's prompt during the forward pass. The paper finds middle layers work best and
reports > 90% wedding-related completions with the weddings vector vs ~2% unsteered, and lower toxicity with a
"Love − Hate" style vector.

**What we measured** (GPT-2-XL, {{n_per_prompt}} samples × 3 prompts per setting):

- Weddings vector: wedding words in {{wedding_base|pct}} of completions unsteered → {{wedding_c4|pct}} at c = 4
  (random vector of the same norm: {{wedding_rand_c4|pct}}); the best coefficient reaches {{wedding_max|pct}}.
- Love − Hate vector: completions judged positive {{love_base|pct}} → {{love_c4|pct}} at c = 4 (random:
  {{love_rand_c4|pct}}).

## Limitations

- Keyword and sentiment-classifier scoring are crude; large coefficients also degrade fluency (see examples).
- Few prompts and samples; results vary with seed and coefficient.
