---
slug: stories-in-space
title: Meandering on manifolds (stories in emotion space)
category: manifold
collection: goodfire-geometry
models: [Qwen2.5-1.5B-Instruct]
fidelity: recomputed
evidence: correlational
order: 650
summary: As Llama reads a story sentence by sentence, its internal state wanders through a map of emotions laid out like human valence and arousal. In Qwen2.5-1.5B the emotional reading is only just decodable, and the geometry matches behaviour weakly.
viz: stories
data: stories-in-space.json
locations:
  - { model: qwen3-1.7b, part: resid, layers: [5, 11], label: "running emotional reading of a story (shown for Qwen2.5-1.5B-Instruct)" }
tags: [in-context learning, emotions, trajectories, belief space]
related: [emotions, manifold-steering, belief-fractal]
quote:
  text: As an LLM reads sentence by sentence, its internal activations move through conceptual space
  source: stories-blog
citations:
  - id: bigelow2026
    title: "Stories in Space: In-Context Learning Trajectories in Conceptual Belief Space"
    authors: Bigelow, Sarfati, Wurgaft, Lewis, McGrath, Merullo, Geiger, Lubana
    year: 2026
    venue: arXiv (CC BY 4.0)
    url: https://arxiv.org/abs/2605.12412
  - id: stories-blog
    title: Meandering on Manifolds (Neural Geometry series)
    authors: Goodfire
    year: 2026
    url: https://www.goodfire.com/research/stories-in-space
  - id: simplestories
    title: SimpleStories dataset
    authors: SimpleStories authors
    year: 2025
    url: https://huggingface.co/datasets/SimpleStories/SimpleStories
  - id: warriner2013
    title: Norms of valence, arousal, and dominance for 13,915 English lemmas
    authors: Warriner, Kuperman, Brysbaert
    year: 2013
    url: https://link.springer.com/article/10.3758/s13428-012-0314-x
replications:
  - who: This atlas
    url: https://github.com/BurnyCoder/circuits-and-manifolds-in-llms-atlas/blob/main/src/atlas/entries/stories_in_space.py
    model: Qwen2.5-1.5B-Instruct
    outcome: ours
    note: Per-sentence rating readout, layer-sweep ridge probes on held-out stories, behaviour–activation distance correlation, valence–arousal trajectories.
---

## In plain words

Reading a story changes how you feel about it, sentence by sentence. Bigelow et al. tracked the same thing inside a
language model: after each sentence they ask the model how happy, sad, angry… the story feels, and they find that the
model's internal activations carry this running reading — the story traces a path through an "emotion space" whose
layout matches how humans place emotions by pleasantness and intensity.

## How it works (technical)

For each story prefix the behavioural readout is the expected 0–9 rating (single-digit answers) from the model's digit probabilities, one query
per emotion. Ridge probes map the prefix's final-token activation to the six ratings; in Llama-3.1-8B-Instruct (layer 9)
the probes reach RMSE 0.09, behaviour and activation distance matrices correlate at r = .92, and the emotions' layout
matches human valence–arousal ratings (r = .93).

**What we measured** ({{n_stories}} SimpleStories, {{n_prefixes}} prefixes): held-out-story probe $R^2$ =
{{best_r2}} at layer {{best_layer}}. Representational similarity across {{n_rsa_prefixes}} prefixes: pairwise distances
between rating vectors and between activations correlate at Spearman r = {{r_geom}}. Shuffling whole stories' ratings
gives at most {{r_geom_null95}} (95th percentile).

**Confirmatory re-test** (pre-stated 2026-11-28; {{f_n_stories}} fresh stories, layer chosen on validation stories,
95% CI from resampling test stories): $R^2$ = {{f_r2}} ({{f_r2_lo}}–{{f_r2_hi}}) at layer {{f_layer}}, so the 0.5 bar is
inside the interval (inconclusive). A bag-of-words model on the same prefix text reaches {{f_bow_r2}}, so the
activations carry more than the words alone. Geometry again agrees only weakly: RSA r = {{f_rsa}} against a
story-shuffle 95th percentile of {{f_rsa_null95}} (above chance, far below the 0.3 bar, fail).

## Limitations

- Smaller instruct model; trajectories use human norms of the six emotion words to place ratings in 2-D, a simplification
  of the paper's UMAP/PCA spaces; steering experiments are not reproduced.
