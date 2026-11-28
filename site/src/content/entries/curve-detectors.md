---
slug: curve-detectors
title: Curve detectors (InceptionV1)
category: neuron
collection: distill
models: [InceptionV1]
fidelity: recomputed
evidence: correlational
order: 50
summary: Ten neurons in an image classifier each detect curves of one orientation — together they cover the whole circle, like a compass rose.
viz: curve-detectors
data: curve-detectors.json
locations:
  - { model: inceptionv1, part: stage, name: mixed3b, label: "10 curve detectors (379, 406, 385, 343, 342, 388, 324, 340, 330, 349)" }
  - { model: inceptionv1, part: stage, name: mixed3a, label: earlier curve family (Cammarata et al.) }
  - { model: inceptionv1, part: stage, name: conv2d2, label: curve precursors (Cammarata et al.) }
tags: [vision, feature visualization, equivariance, Distill]
related: [block-sparse-featurizers]
citations:
  - id: cammarata2020
    title: Curve Detectors
    authors: Cammarata, Goh, Carter, Schubert, Petrov, Olah
    year: 2020
    venue: Distill
    url: https://distill.pub/2020/circuits/curve-detectors/
  - id: olah2020zoom
    title: "Zoom In: An Introduction to Circuits"
    authors: Olah, Cammarata, Schubert, Goh, Petrov, Carter
    year: 2020
    venue: Distill
    url: https://distill.pub/2020/circuits/zoom-in/
  - id: szegedy2015
    title: Going Deeper with Convolutions (InceptionV1 / GoogLeNet)
    authors: Szegedy, Liu, Jia, Sermanet, Reed, Anguelov, Erhan, Vanhoucke, Rabinovich
    year: 2015
    venue: CVPR 2015
    url: https://arxiv.org/abs/1409.4842
  - id: lucent
    title: "Lucent: Lucid library adapted for PyTorch"
    authors: Lim et al.
    year: 2020
    url: https://github.com/greentfrapp/lucent
replications:
  - who: This atlas
    url: https://github.com/BurnyCoder/circuits-and-manifolds-from-mechanistic-interpretability-in-llms-and-other-models-atlas/blob/main/src/atlas/entries/curve_detectors.py
    model: InceptionV1 (Lucid weights via Lucent)
    outcome: ours
    note: Synthetic arcs at 72 orientations × 4 radii plus straight lines, 20 random control neurons, and our own feature visualizations.
---

## In plain words

Inside InceptionV1, a 2014 image classifier, the layer called mixed3b contains a family of about ten neurons that each
fire for a curve bending in one particular direction — one for curves opening upward, one for curves opening to the
right, and so on. Nobody programmed them; they emerged from training on photos. Cammarata et al. checked this from many
angles: feature visualization, dataset examples, and carefully drawn synthetic curves. Here we redo the synthetic test:
draw arcs at every orientation and watch which neuron lights up.

## How it works (technical)

Each stimulus is a black arc on white whose midpoint sits at the image center and whose bulge points in direction
$\theta$ (radii 15–110 px), plus straight lines. We read each neuron's activation at the center of the 28×28 mixed3b map
(max over the central 3×3) and summarize its orientation tuning at its best radius with the selectivity
$S = |\sum_\theta r_\theta e^{i\theta}| / \sum_\theta r_\theta$ on rectified responses (1 = fires for a single angle,
0 = equally for all).

**What we measured:**

- Median selectivity {{sel_curve_median}} for the ten curve neurons versus {{sel_control_median}} for
  {{n_controls}} random other mixed3b neurons ({{sel_control_resp_median}} over the {{n_controls_resp}} controls that
  respond at all; 95% CI of the difference {{sel_diff_lo}}–{{sel_diff_hi}}).
- {{n_curve_gt_line}} of 10 respond more than 1.5× as strongly to their best curve as to any straight line.
- Their preferred orientations tile the circle: the largest gap between neighbours is {{max_orient_gap}}°.

## Limitations

- Synthetic stimuli test only one aspect; the original article also uses dataset examples, attribution and a human
  comparison, and the follow-up "Curve Circuits" explains how the detectors are built from earlier layers.
- Feature visualizations depend on optimization settings (Lucent defaults, 512 steps).
- Reading only the central position ignores the (translation-equivariant) responses elsewhere in the map.
