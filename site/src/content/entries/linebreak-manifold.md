---
slug: linebreak-manifold
title: The character-counting manifold (line breaking)
category: manifold
collection: anthropic-tc
models: [Qwen3-1.7B, GPT-2 small]
fidelity: replicated
evidence: correlational
order: 120
summary: Claude tracks how many characters it has written on a twisting, rippled curve to know when to break a line. In Qwen3-1.7B (and GPT-2) the count is decodable (whether above the paper-level bar is inconclusive on fresh data), but the curve is far less compact.
viz: linebreak
data: linebreak-manifold.json
locations:
  - { model: qwen3-1.7b, part: resid, layers: [1, 12], label: character count since the last newline (probe + curved manifold) }
  - { model: claude-3.5-haiku, part: band, depth: [0, 15], label: count manifold built by early attention heads; boundary heads rotate it (Gurnee et al.) }
tags: [counting, manifolds, ringing, place cells, QK rotation]
related: [othello-gpt, calendar-circles, sae-manifolds]
quote:
  text: "[C]haracter counts are represented on low-dimensional curved manifolds"
  source: gurnee2025
  where: Abstract of the arXiv version (2601.04480)
citations:
  - id: gurnee2025
    title: "When Models Manipulate Manifolds: The Geometry of a Counting Task"
    authors: Gurnee, Ameisen, Kauvar, Tarng, Pearce, Olah, Batson
    year: 2025
    venue: Transformer Circuits Thread (arXiv 2601.04480)
    url: https://transformer-circuits.pub/2025/linebreaks/index.html
  - id: sinii2026
    title: Chasing the Counting Manifold in Open LLMs
    authors: Sinii, Balagansky (T-Tech)
    year: 2026
    url: https://t-tech-manifolds.hf.space/
  - id: ttech-code
    title: counting_manifolds (code, Apache-2.0)
    authors: corl-team
    year: 2026
    url: https://github.com/corl-team/counting_manifolds
replications:
  - who: This atlas
    url: https://github.com/BurnyCoder/circuits-and-manifolds-from-mechanistic-interpretability-in-llms-and-other-models-atlas/blob/main/src/atlas/entries/linebreak_manifold.py
    model: Qwen3-1.7B (GPT-2 small control)
    outcome: ours
    note: Probes by layer with a shuffled-label control, per-count mean geometry, cosine ringing, and newline-prediction AUC.
  - who: Sinii & Balagansky (T-Tech) — "Chasing the Counting Manifold in Open LLMs"
    url: https://t-tech-manifolds.hf.space/
    model: GPT-2, Pythia, Gemma-2/3, Qwen3 0.6B–8B, Llama-3.1-8B
    outcome: partial
    note: Linear decodability (R² up to 0.87) and a helix with period ≈ 155 characters replicate; Anthropic's very-early-layer peak does not; no ablations or QK-twisting analysis.
  - who: statescope — counting pathway in a hybrid model
    url: https://github.com/mconcat/statescope
    model: Qwen3.5-0.8B (GatedDeltaNet hybrid)
    outcome: partial
    note: Finds a causal counting pathway through a recurrent head; does not claim the manifold geometry.
---

## In plain words

When text is hard-wrapped at, say, 80 characters, a model predicting the next token must notice when the line is
almost full and emit a newline. Anthropic found that Claude 3.5 Haiku keeps a running **character count** — not on a
straight number line, but on a curve that twists through about six dimensions, with sparse "place-cell"-like features
tiling it. Special attention heads then rotate this curve so it lines up with the line width and the model can see
how much room is left. Claude is closed, so here we look for the same counting manifold in open models.

## How it works (technical)

Gurnee et al. average Haiku's layer-2 residual stream for each count (1–150): the top 6 principal components hold 95% of
the variance, the curve is helix-like, and the cosine-similarity matrix of the means "rings" (a diagonal band plus
echoes) — the signature of a rippled curve that trades dimension for distinguishability. A linear probe after layer 1
reaches $R^2$ = 0.985.

**What we measured** (Qwen3-1.7B, {{q_tokens}} tokens of WikiText wrapped at 40–150 characters):

- A ridge probe reads the count from the residual stream with held-out $R^2$ = {{q_best_r2}} at layer {{q_best_layer}}
  (shuffled labels: {{q_best_r2_shuffled}}); the best value within layers 0–2 is {{q_early_r2}}. GPT-2 small peaks at
  {{g_best_r2}}.
- The per-count mean activations at layer {{q_best_layer}} put {{q_var6|pct}} of their variance in 6 principal components
  ({{q_var3|pct}} in 3), and the PC1–PC2 angle turns {{q_deg_per_char}}° per character.
- Qwen3-1.7B predicts real line breaks well: the AUC of its newline probability is {{q_auc}} (GPT-2 small {{g_auc}}).
- Confirmatory re-test (pre-stated 2026-11-28) on fresh documents, with the layer chosen on validation documents and
  95% CIs from resampling test documents: $R^2$ = {{qf_r2}} ({{qf_r2_lo}}–{{qf_r2_hi}}) at layer {{qf_layer}}, so the
  0.7 bar is inside the interval (inconclusive). The best early layer ({{qf_early_layer}}) reaches {{qf_ratio}} of that
  ({{qf_ratio_lo}}–{{qf_ratio_hi}}) against a 0.9 bar (inconclusive). GPT-2 small reaches {{gf_r2}}
  ({{gf_r2_lo}}–{{gf_r2_hi}}): the count is just as readable in the much weaker model.

## Limitations

- Correlational only: we do not ablate or patch the count subspace, and we do not test the boundary-head "QK
  twisting" (no open replication of that part exists yet).
- WikiText paragraphs wrapped with Python's `textwrap`, not the paper's data; counts above ~120 are rarer.
- Probes see a subsample of tokens; the manifold uses per-count means over all tokens.
