---
slug: colour-bicone
title: The color bicone in an image autoencoder
category: manifold
collection: other
models: [FLUX VAE]
fidelity: recomputed
evidence: correlational
order: 140
summary: Inside the FLUX image model's autoencoder, every solid color lands on a 3-D double cone — lightness up the middle, hue around, saturation outward.
viz: colour-bicone
data: colour-bicone.json
locations:
  - { model: flux-vae, part: stage, name: latent, label: 3-D hue–saturation–lightness subspace of the latent }
tags: [vision, color, VAE, diffusion, PCA]
related: [calendar-circles, block-sparse-featurizers]
citations:
  - id: pach2026
    title: "The Latent Color Subspace: Emergent Order in High-Dimensional Chaos"
    authors: Pach, Bader, Bouniot, Belongie, Akata
    year: 2026
    venue: ICML 2026
    url: https://arxiv.org/abs/2603.12261
  - id: lcs-code
    title: LCS (official code)
    authors: ExplainableML
    year: 2026
    url: https://github.com/ExplainableML/LCS
  - id: flux-schnell
    title: FLUX.1 [schnell] (model card, Apache-2.0)
    authors: Black Forest Labs
    year: 2024
    url: https://huggingface.co/black-forest-labs/FLUX.1-schnell
  - id: flux-vae
    title: diffusers/FLUX.1-vae (FLUX.1 autoencoder in diffusers format)
    authors: Hugging Face diffusers team
    year: 2025
    url: https://huggingface.co/diffusers/FLUX.1-vae
replications:
  - who: This atlas
    url: https://github.com/BurnyCoder/circuits-and-manifolds-in-llms-atlas/blob/main/src/atlas/entries/colour_bicone.py
    model: FLUX.1 VAE (diffusers/FLUX.1-vae)
    outcome: ours
    note: Solid-color PCA with a random-noise control and correlations of each axis with lightness, hue and chroma.
---

## In plain words

Image generators like FLUX first compress pictures into a smaller "latent" code with an autoencoder. Pach et al. asked
where color lives in that code. Feed in plain one-color images and look at the codes: they all fall into a tiny 3-D
corner of the 16-dimensional space, arranged exactly like the textbook **HSL color solid** — black at one tip, white at
the other, grays along the axis, and the rainbow of hues around the middle, most saturated at the widest point. Knowing
this, you can recolor generated images by moving latents in closed form.

## How it works (technical)

Each solid-color image is encoded and its latent averaged over space, then centered and decomposed with PCA. The first
three components explain essentially all of the variance; PC1 orders colors by lightness, the angle in the PC2–PC3
plane by hue, and the distance from the axis by saturation (shrinking toward black and white: a bicone). The paper then
tracks these coordinates through the diffusion process and edits colors in the latent space.

**What we measured** ({{n_colors}} colors sampled uniformly in HSV; {{latent_channels}}-channel latent):

- The top three principal components explain {{var3|pct}} of the variance (PC1 alone {{var1|pct}}); for random-noise
  images the same pipeline gives {{var_noise3|pct}}.
- Correlation of PC1 with lightness {{r_light}}; circular correlation of the PC2–PC3 angle with hue {{r_hue}}; correlation
  of the radius with chroma {{r_sat}}.

## Limitations

- We use the FLUX.1 autoencoder from `diffusers/FLUX.1-vae` (ungated; same configuration as the FLUX.1 VAE); the paper
  used FLUX.1-dev's VAE from the gated official repository. We only reproduce the
  discovery step, not the denoising-trajectory analysis or the color edits.
- Solid colors are the easiest case; textures and lighting are not tested.
