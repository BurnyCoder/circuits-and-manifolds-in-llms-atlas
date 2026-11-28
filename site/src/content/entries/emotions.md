---
slug: emotions
title: Emotion vectors (open model)
category: steering
collection: anthropic-tc
models: [Qwen3-1.7B]
fidelity: replicated
evidence: mixed
order: 420
summary: Anthropic found emotion vectors in Claude laid out by pleasantness and intensity, and steering with them changed its choices. In Qwen3-1.7B we recover the pleasantness axis strongly, but not the intensity axis or the effect on choices.
viz: emotions
data: emotions.json
locations:
  - { model: qwen3-1.7b, part: resid, layer: 17, label: emotion vectors at ~2/3 depth (valence / arousal map, dose sweep, preference steering) }
  - { model: claude-sonnet-4.5, part: band, depth: [60, 72], label: 171 emotion vectors at ~2/3 depth drive preferences and misaligned behaviour (Sofroniew et al.) }
tags: [emotions, valence, arousal, steering, functional emotions]
related: [emotions-claude, stories-in-space, refusal-direction]
quote:
  text: do not imply that LLMs have any subjective experience of emotions
  source: sofroniew2026
  where: Abstract
citations:
  - id: sofroniew2026
    title: Emotion Concepts and their Function in a Large Language Model
    authors: Sofroniew, Kauvar, Saunders, Chen, Henighan, … Olah, Lindsey
    year: 2026
    venue: Transformer Circuits Thread (arXiv 2604.07729)
    url: https://transformer-circuits.pub/2026/emotions/index.html
  - id: warriner2013
    title: Norms of valence, arousal, and dominance for 13,915 English lemmas
    authors: Warriner, Kuperman, Brysbaert
    year: 2013
    venue: Behavior Research Methods
    url: https://link.springer.com/article/10.3758/s13428-012-0314-x
replications:
  - who: This atlas
    url: https://github.com/BurnyCoder/circuits-and-manifolds-in-llms-atlas/blob/main/src/atlas/entries/emotions.py
    model: Qwen3-1.7B (writes its own stories)
    outcome: ours
    note: 80 emotions with human norms, neutral-PC projection, shuffled-label control, dose sweep and a preference-steering test with a random-vector control.
  - who: Hong — replication on Llama-3.1-8B-Instruct
    url: https://jyhong.gitlab.io/post/emotion-vector-part1/
    model: Llama-3.1-8B-Instruct (full 205k-story corpus)
    outcome: partial
    note: 10 of 11 checks pass (steering-vs-correlation r = 0.956); the implicit-emotion test fails and Elo effects are far smaller.
  - who: Hollowell — emotion probes in Gemma-2-27B
    url: https://huggingface.co/papers/2609.22208
    model: Gemma-2-27B base
    outcome: partial
    note: PC1 26.7% with valence r = 0.72; arousal r = 0.67 but not counted as replicated; observational only.
  - who: Jain et al. — emotion vectors and risky behaviour
    url: https://lesswrong.com/posts/PYcqf9DBMGjTyJSjg/model-s-emotions-noticeably-change-when-they-take-risky-1
    model: Gemma-3 4B/27B, GPT-OSS-120B, GLM
    outcome: partial
    note: Steering "excited" cuts reward hacking by ~40 points, but steering "calm" often increased hacking — the opposite of Anthropic's result.
---

## In plain words

Anthropic found that Claude represents emotion concepts — fear, calm, joy, desperation — as directions in its activations.
These "functional emotions" are organised like human ones (the biggest axis is pleasant vs unpleasant, the second is calm
vs excited), they light up when a situation calls for them (a dangerous medicine dose raises "afraid"), and adding them
changes behaviour. That does not mean the model feels anything. Here we rebuild a small version on an open model.

## How it works (technical)

Each emotion vector is the mean residual activation while the model reads stories about that emotion, minus the mean
over all emotions, with the top principal components of emotionally neutral text projected out (confound removal). In
Claude Sonnet 4.5 PC1 (26%) correlates with human valence (r = 0.81) and PC2 (15%) with arousal (r = 0.66); steering at
strength 0.5 of the residual norm shifts activity preferences (blissful +212 Elo, hostile −303).

**What we measured** (Qwen3-1.7B layer {{layer}}, {{n_emotions}} emotions, {{n_stories}} stories it wrote itself):

- PC1 explains {{var_pc1|pct}} of the variance and correlates with human valence at |r| = {{r_valence}} (shuffled labels,
  95th percentile: {{null_valence_p95}}); PC2 ({{var_pc2|pct}}) correlates with human arousal at |r| = {{r_arousal}}.
- Tylenol sweep: per tenfold increase in dose, the "afraid" projection changes by {{afraid_slope}} and "calm" by
  {{calm_slope}}.
- Preference test: the probability of choosing option A is {{pref_base}} at baseline, {{pref_blissful}} with the blissful
  vector added to A's words, {{pref_hostile}} with hostile, and {{pref_random}} with a random vector.

## Limitations

- A 1.7B model writing its own short stories is far noisier than Claude's 205k-story corpus; 80 emotions instead of 171.
- The preference test uses benign everyday activities; the paper's misalignment experiments (blackmail, reward hacking)
  are not meaningful at this scale.
- Functional emotion representations are not evidence of subjective experience (the authors stress this too).
