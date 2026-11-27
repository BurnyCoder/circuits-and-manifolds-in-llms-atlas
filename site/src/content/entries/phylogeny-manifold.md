---
slug: phylogeny-manifold
title: The tree of life inside a DNA model (Evo 2)
category: manifold
collection: goodfire-geometry
models: [Evo 2]
fidelity: redrawn
evidence: correlational
order: 660
summary: A genome language model arranges 2,400+ bacterial species so that distances along a curved manifold in its activations follow the tree of life.
viz: phylogeny
locations:
  - { model: evo2, part: resid, layer: 24, label: phylogeny encoded as geodesic distance on a curved manifold (layer 24 of 32) }
tags: [biology, phylogeny, manifolds, geodesics, DNA models]
related: [world-model, calendar-circles, colour-bicone]
citations:
  - id: pearce2025
    title: Finding the Tree of Life in Evo 2
    authors: Pearce, Simon, Byun, Balsam (Goodfire)
    year: 2025
    url: https://www.goodfire.com/research/phylogeny-manifold
  - id: geiger2026
    title: The World Inside Neural Networks
    authors: Geiger, Lubana, Fel, Merullo, Byun, Lewis, McGrath (Goodfire)
    year: 2026
    url: https://www.goodfire.com/research/the-world-inside-neural-networks
  - id: gtdb
    title: Genome Taxonomy Database (GTDB)
    authors: Parks et al.
    year: 2022
    url: https://gtdb.ecogenomic.org/
---

## In plain words

Evo 2 is a language model trained on DNA. Goodfire fed it chunks of genomes from more than 2,400 bacterial species and
asked whether its internal representations know how species are related. They do — but not with straight-line distances.
Species sit on a curved surface, and the distance measured *along* that surface tracks the evolutionary tree, much like
the distance between two cities measured along the Earth's surface rather than through it.

## How it works (technical)

Embeddings are averaged over 4-kbp genome regions at layer 24 of 32. A k-nearest-neighbour graph (K = 27, angular
distance) gives geodesic distances that correlate roughly linearly with phylogenetic branch-length distance (GTDB, 120
marker genes); Mycoplasmatales are the outlier. A learned flat ~10-dimensional "phylogeny subspace" predicts distances
with r = 0.98 on training clades and ≈ 0.9 on held-out clades, capturing ~70% of variance — "flat structure plus
ripples". The analysis is correlational (no interventions). The visualization is a simulation of the geodesic-vs-straight
idea on a toy tree.

## Limitations

- No code or data released; model size used is not stated; correlational only.
