---
slug: sae-manifolds
title: Can SAEs capture neural geometry?
category: feature
collection: goodfire-geometry
models: [TopK SAEs on synthetic manifolds (trained here)]
fidelity: recomputed
evidence: correlational
order: 610
summary: Sparse autoencoders chop curved concepts into pieces — "capture" only works when the sparsity matches the data; otherwise manifolds shatter or dilute.
viz: sae-manifolds
data: sae-manifolds.json
tags: [SAE, manifolds, shattering, dilution, synthetic benchmark]
related: [calendar-circles, block-sparse-featurizers, toy-superposition]
quote:
  text: When we train SAEs on actual neural network representations, we observe dilution.
  source: bhalla2026
citations:
  - id: bhalla2026
    title: Do Sparse Autoencoders Capture Concept Manifolds?
    authors: Bhalla, Fel, Rager, Feucht, Haklay, Wurgaft, et al. (Goodfire)
    year: 2026
    venue: arXiv (CC BY 4.0)
    url: https://arxiv.org/abs/2604.28119
  - id: bhalla-blog
    title: Can SAEs Capture Neural Geometry? (Neural Geometry series)
    authors: Goodfire
    year: 2026
    url: https://www.goodfire.com/research/can-saes-capture-neural-geometry
  - id: sae-manifold-code
    title: sae-manifold (code, MIT)
    authors: Goodfire
    year: 2026
    url: https://github.com/goodfire-ai/sae-manifold
replications:
  - who: This atlas
    url: https://github.com/BurnyCoder/circuits-and-manifolds-in-llms-atlas/blob/main/src/atlas/entries/sae_manifolds.py
    model: TopK SAEs on 16 synthetic manifolds
    outcome: ours
    note: Re-implemented synthetic benchmark (the official repo has no generator); simplified regime classification stated in the text.
---

## In plain words

Sparse autoencoders (SAEs) break a model's activations into many single directions ("features"). But many concepts are
curved — days of the week on a circle, colors on a cone. What does an SAE do with a circle? Goodfire built a clean test:
hide circles, spheres, tori and Möbius strips in a 128-dimensional space, mix several at once, train SAEs, and look. They
find three outcomes: **capture** (a manifold gets about as many atoms as its dimensions), **shattering** (many atoms,
each covering a little patch — like place cells), and **dilution** (many broad, overlapping atoms). Here we rebuild the
synthetic benchmark.

## How it works (technical)

Each manifold instance is centered, scaled to unit RMS and embedded by a random orthonormal map into $d = 128$; each
sample sums $L_0 = 4$ random manifolds at random points; a TopK SAE (512 unit-norm atoms) is trained for a sweep of $k$.
Following the paper's appendix, everything is measured on the superposed inputs, on the rows where a manifold is
present:
- **Restricted $R^2$** (the paper's capture score): can $k_i$ decoder atoms rebuild that manifold's contribution?
- **Support size** (atoms firing on ≥ 10% of its rows).
- **Receptive-field spread** (how far apart the points an atom fires on are, relative to the manifold).

**What we measured** ({{n_instances}} manifold instances, {{steps}} training steps per SAE):

- Mean restricted $R^2$ peaks at k = {{k_best_r2}} ({{r2_best}}). It is {{r2_min_k}} at the smallest k and {{r2_max_k}} at
  k = {{k_max}}. At most {{capture_max|pct}} of manifolds get the "capture" label.
- Shattering goes from {{shatter_at_min_k|pct}} at the smallest k to {{shatter_at_max_k|pct}} at k = {{k_max}}. Dilution goes
  from {{dilution_at_min_k|pct}} to {{dilution_at_max_k|pct}}.

## Limitations

- Our regime rule (support size and spread) is a simplification of the paper's Ising-coupling analysis.
- One SAE type (TopK). The paper's LLM experiments (Llama-3.1-8B, five SAE families) are not
  reproduced — see the calendar-circles entry for how GPT-2's SAE latents tile the day circle.
