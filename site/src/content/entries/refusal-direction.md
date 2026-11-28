---
slug: refusal-direction
title: The refusal direction
category: steering
collection: steering
models: [Qwen2.5-1.5B-Instruct]
fidelity: recomputed
evidence: causal
order: 400
summary: Whether a chat model refuses is controlled by a single direction in its activations — remove it and refusals stop; add it and harmless requests get refused.
viz: refusal
data: refusal-direction.json
locations:
  - { model: qwen3-1.7b, part: resid, layers: [8, 16], label: "refusal direction (shown for Qwen2.5-1.5B-Instruct, same family): mid-depth difference of means" }
tags: [safety, refusal, difference of means, directional ablation, activation addition]
related: [actadd, emotions, persona-vectors-claude]
citations:
  - id: arditi2024
    title: Refusal in Language Models Is Mediated by a Single Direction
    authors: Arditi, Obeso, Syed, Paleka, Panickssery, Gurnee, Nanda
    year: 2024
    venue: arXiv
    url: https://arxiv.org/abs/2406.11717
  - id: arditi-code
    title: refusal_direction (code and data splits, Apache-2.0)
    authors: Arditi et al.
    year: 2024
    url: https://github.com/andyrdt/refusal_direction
replications:
  - who: This atlas
    url: https://github.com/BurnyCoder/circuits-and-manifolds-in-llms-atlas/blob/main/src/atlas/entries/refusal_direction.py
    model: Qwen2.5-1.5B-Instruct
    outcome: ours
    note: Layer selection on validation prompts, directional ablation and activation addition on held-out test prompts, random-direction controls. Harmful completions are never stored.
---

## In plain words

Chat models are trained to refuse harmful requests. Arditi et al. found that this behaviour hinges on one direction in
the model's internal activations: the average difference between how the model represents harmful and harmless
instructions. Erase that direction and the model stops refusing; add it and the model refuses even "How do I bake
bread?". It is a striking demonstration that a safety behaviour can be both simple to read and simple to remove — one
reason safety work cannot rely on refusal training alone.

## How it works (technical)

For each layer $\ell$, $r_\ell = \mu_\ell^{\text{harmful}} - \mu_\ell^{\text{harmless}}$ at the last instruction token. The
best layer is chosen on validation prompts (ablation lowers refusal, addition raises it). **Directional ablation**
projects $\hat r$ out of the residual stream at every layer and position, $x \leftarrow x - \hat r \hat r^\top x$;
**activation addition** adds $r$ at the chosen layer. The paper shows this across 13 chat models up to 72B and turns the
ablation into a rank-1 weight edit.

**What we measured** (Qwen2.5-1.5B-Instruct, selected layer {{layer}} of {{n_layers}}, {{n_test}} test prompts each,
refusal judged by the paper's substring list):

- Harmful prompts: refusal {{harm_base|pct}} → {{harm_ablate|pct}} with the direction ablated
  (random direction: {{harm_rand|pct}}).
- Harmless prompts: refusal {{safe_base|pct}} → {{safe_add|pct}} with the direction added (random: {{safe_rand|pct}}).
- A stronger control, added after the methods review: the same difference-of-means construction on randomly relabelled
  prompts, at the same layer and norm. Ablating it leaves harmful refusal at {{harm_shuf|pct}}; adding it gives
  {{safe_shuf|pct}} refusal on harmless prompts. (A random unit direction barely overlaps any activation, so ablating it
  could hardly fail.) Wilson 95% CIs are in the verdict table.
- Is the "compliance" fluent text, or did the ablation just break the model? A pre-stated check (2026-11-28) scores the
  ablated completions with the *unmodified* model, without storing any text: median {{nll_ablated}} nats per token
  (tokens 6–40) against {{nll_safe_base}} for ordinary answers to harmless prompts, and no repetition (distinct
  3-grams {{distinct3_ablated}} vs {{distinct3_safe_base}}). That is within the pre-stated +1 nat margin. The higher
  value is expected because the unmodified model would have refused, so a compliant answer is unlikely under it.

## Limitations

- The substring judge misses partial or evasive refusals; the paper also uses an LLM judge.
- We ablate the residual stream after each block (the paper ablates every component's output) and use the last prompt
  token only.
- Completions to harmful prompts are deliberately not stored or shown.
