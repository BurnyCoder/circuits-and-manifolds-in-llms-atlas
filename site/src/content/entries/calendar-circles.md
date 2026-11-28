---
slug: calendar-circles
title: Calendar circles (days and months)
category: manifold
collection: tegmark
models: [GPT-2 small, Qwen3-1.7B]
fidelity: recomputed
evidence: mixed
order: 100
summary: Days of the week and months of the year are stored on circles — and moving a day along its circle changes the model's date arithmetic.
viz: calendar-circles
data: calendar-circles.json
locations:
  - { model: gpt2-small, part: resid, layer: 6, label: day / month / year circles (Engels et al.; residual entering block 7) }
  - { model: qwen3-1.7b, part: resid, layers: [2, 20], label: calendar circles across the layer sweep; causal patch tested per layer }
  - { model: llama-3.1-8b, part: resid, layer: 28, label: weekday manifold used for manifold steering (Wurgaft et al. 2026) }
tags: [multi-dimensional features, circular features, SAE, intervention]
related: [number-helix, grokking, manifold-steering, geometric-calculator]
citations:
  - id: engels2024
    title: Not All Language Model Features Are One-Dimensionally Linear
    authors: Engels, Michaud, Liao, Gurnee, Tegmark
    year: 2024
    venue: ICLR 2025
    url: https://arxiv.org/abs/2405.14860
  - id: engels-code
    title: MultiDimensionalFeatures (official code, MIT)
    authors: Engels et al.
    year: 2024
    url: https://github.com/JoshEngels/MultiDimensionalFeatures
  - id: bloom2024
    title: GPT-2 small residual-stream SAEs (gpt2-small-res-jb, via SAELens)
    authors: Bloom
    year: 2024
    url: https://decoderesearch.github.io/SAELens/latest/pretrained_saes/gpt2-small/
  - id: wurgaft2026
    title: Manifold Steering Reveals the Shared Geometry of Neural Network Representation and Behavior
    authors: Wurgaft, Rager, Kowal, et al.
    year: 2026
    url: https://arxiv.org/abs/2605.05115
  - id: marbut2026
    title: CLT features sharpen the cyclical day-of-week manifold in Gemma-2-2B
    authors: Marbut
    year: 2026
    venue: LessWrong
    url: https://www.lesswrong.com/posts/vwdHTmorDPFB8vFQq/clt-features-sharpen-the-cyclical-day-of-week-manifold-in
replications:
  - who: This atlas
    url: https://github.com/BurnyCoder/circuits-and-manifolds-in-llms-atlas/blob/main/src/atlas/entries/calendar_circles.py
    model: GPT-2 small (raw + SAE) and Qwen3-1.7B (layer sweep + causal patch)
    outcome: ours
    note: Cyclic-order permutation tests against random-word controls, SAE latent tiling, and circle-plane patching with a random-plane control.
  - who: Marbut — day-of-week circle in Gemma-2-2B transcoder features
    url: https://www.lesswrong.com/posts/vwdHTmorDPFB8vFQq/clt-features-sharpen-the-cyclical-day-of-week-manifold-in
    model: Gemma-2-2B + cross-layer transcoder (layer 20)
    outcome: replicated
    note: Circle appears in activation, behaviour and transcoder space; the "cleaner in CLT space" advantage mostly vanishes after mean-centering.
  - who: Gupta, Fruengel & Heimersheim — "Spokes, not a Ring"
    url: https://arxiv.org/abs/2610.37796
    model: Llama-3.1-8B, Mistral-7B, GPT-2 XL
    outcome: partial
    note: Steering along the weekday circle is step-like at early layers (positions between days are not used); smooth behaviour appears mainly late.
---

## In plain words

You might expect a language model to store "Monday", "Tuesday", … as unrelated items. Engels et al. found something
more structured: the representations of the seven days sit on a **circle**, in calendar order, so that Sunday wraps
back around to Monday; months do the same with twelve points. A circle is the natural shape for things that repeat,
and models seem to discover it on their own. In bigger models the circle is not just decoration: when the model does
"two days from Saturday", moving the "Saturday" activation around the circle to "Monday" changes the answer.

## How it works (technical)

Engels et al. define *irreducible multi-dimensional features* — representations that cannot be split into independent
1-D pieces — and find them by clustering sparse-autoencoder decoder directions; reconstructing activations from one
cluster and projecting on its top principal components reveals circles for days, months and years (GPT-2 small layer 7,
Mistral 7B). Interventions with circular probes on the top-5 PCs shift Mistral 7B and Llama 3 8B's answers in
modular day/month arithmetic.

**What we measured:**

- GPT-2 small, layer 7 (template means removed): the day means lie in cyclic order with score {{g2_days_score}}
  (1 = perfect, permutation p = {{g2_days_p}}); months score {{g2_months_score}} (p = {{g2_months_p}}). Random sets of
  ordinary words (from a pool of 93) look cyclic at p < 0.05 in {{ctrl_sig_rate|pct}} of {{n_ctrl}} draws, and days and
  months concentrate more of their class-mean variance in two components ({{g2_days_var2|pct}}, {{g2_months_var2|pct}})
  than 95% of random word sets ({{ctrl_var2_p95_7|pct}} for 7 words, {{ctrl_var2_p95_12|pct}} for 12).
- Reconstructing day activations from the {{g2_sae_days_n}} SAE latents that fire most on days gives cyclic order
  {{g2_sae_days_score}} (months {{g2_sae_months_score}}); individual latents prefer runs of neighbouring days.
- Qwen3-1.7B: the clearest day circle is at layer {{q_days_best_layer}} (score {{q_days_best_score}}), months at layer
  {{q_months_best_layer}} ({{q_months_best_score}}). It answers {{q_acc|pct}} of day-arithmetic prompts correctly.
- Causal test at layer {{q_causal_best_layer}}: moving the start day's activation along the circle plane to another
  day's position makes the answer move to the corresponding day in {{q_causal_best|pct}} of {{q_causal_n}} prompts,
  versus {{q_causal_best_rand|pct}} for a move of the same size in a random plane (norm-matched after the methods review;
the causal row is post-hoc because its templates changed after the smoke test).

## Limitations

- We approximate the paper's SAE-cluster method by taking the latents most active on the target tokens.
- Template-mean removal is our choice for isolating "which day"; it is a supervised step.
- GPT-2 small cannot do day arithmetic (Engels et al.: 8/49), so the causal test uses Qwen3-1.7B; the best intervention
  layer is chosen from a sweep (reported in full in the visualization).
