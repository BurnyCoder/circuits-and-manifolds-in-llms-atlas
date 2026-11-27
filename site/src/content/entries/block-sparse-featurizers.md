---
slug: block-sparse-featurizers
title: Block-sparse featurizers (multi-dimensional concepts)
category: feature
collection: goodfire-geometry
models: [BSF / TopK SAE (trained here), InceptionV1]
fidelity: recomputed
evidence: correlational
order: 620
summary: Many concepts are not single directions but small 2–4-dimensional subspaces; grouping SAE latents into blocks captures them whole.
viz: block-sparse
data: block-sparse-featurizers.json
locations:
  - { model: inceptionv1, part: stage, name: mixed3b, label: one block of curve responses covers all orientations (Fel et al. 2026) }
tags: [block sparsity, SAE, vision, curve detectors, multidimensional features]
related: [curve-detectors, sae-manifolds, colour-bicone]
quote:
  text: most concepts appear to be multidimensional – and specifically, two- to four-dimensional
  source: fel2026
citations:
  - id: fel2026
    title: "Structuring Sparsity: Block-Sparse Featurizers Capture Visual Concept Manifolds"
    authors: Fel, Kowal, Jacobs, Hazra, Bhalla, et al. (Goodfire)
    year: 2026
    venue: arXiv (CC BY 4.0)
    url: https://arxiv.org/abs/2606.25234
  - id: bsf-blog
    title: Block-Sparse Featurizers (Neural Geometry series)
    authors: Goodfire
    year: 2026
    url: https://www.goodfire.com/research/bsf-vision
  - id: bsf-code
    title: block-sparse-featurizer (code, MIT)
    authors: Goodfire
    year: 2026
    url: https://github.com/goodfire-ai/block-sparse-featurizer
replications:
  - who: This atlas
    url: https://github.com/BurnyCoder/circuits-and-manifolds-in-llms-atlas/blob/main/src/atlas/entries/block_sparse.py
    model: toy concept manifolds; InceptionV1 mixed3b
    outcome: ours
    note: BSF vs TopK SAE on sphere-valued toy concepts; harmonic analysis of the curve-response block (top-16 PCs as a stand-in for a learned block).
---

## In plain words

Standard sparse autoencoders assume each concept is one direction that is either on or off. Fel et al. argue that most
visual concepts are small subspaces instead — "where on the hand" or "which orientation" is a position inside the
concept, not a separate concept. A block-sparse featurizer groups latents into blocks and switches whole blocks on or
off, so one block can hold a whole little manifold. In InceptionV1 a single block covers every curve orientation, the
same family of curve detectors described by the Distill Circuits thread.

## How it works (technical)

$z = \Pi_k(xW + b)$ keeps the top-$k$ blocks of size $b$ by $\lVert z_g\rVert_2$ and reconstructs $\hat x = \sum_g z_g D_g$, with
no ReLU inside blocks (a ReLU would restrict each block to a cone). In the paper's toy setting a block-sparse featurizer
recovers each concept with per-block $R^2$ 0.93–0.97 vs ≈ 0.53 for a TopK SAE; on DINOv3 the description length is
minimized at block size 2–4.

**What we measured:**

- Toy ({{n_concepts}} concepts, each a sphere in its own 3-D subspace): median per-concept $R^2$ {{bsf_r2_median}}
  (block-sparse) vs {{sae_r2_median}} (TopK SAE with the same latent budget).
- InceptionV1: in the top-16 principal components of mixed3b's responses to curves, orientation harmonic 1 explains
  {{curve_h1|pct}} and harmonics 1–3 together {{curve_h123|pct}} (shuffled-orientation 95th percentile {{curve_null_p95|pct}}).

## Limitations

- Scaled-down toy; our "curve block" is a PCA stand-in for a learned block; DINOv3 and SDXL experiments are not reproduced.
