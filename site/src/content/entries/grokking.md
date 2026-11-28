---
slug: grokking
title: Grokking modular addition (Fourier multiplication)
category: circuit
collection: other
models: [1-layer transformer (trained here)]
fidelity: recomputed
evidence: causal
order: 30
summary: A tiny transformer memorizes (a + b) mod 113, then suddenly generalizes by placing numbers on circles and adding angles with trig identities.
viz: grokking
data: grokking.json
extra: grokking-runs.json
locations:
  - { model: modadd-1l, part: embed, label: numbers placed on circles at a few key frequencies }
  - { model: modadd-1l, part: attn, layer: 0, label: attention combines the a and b waves }
  - { model: modadd-1l, part: mlp, layer: 0, label: periodic neurons compute cos/sin(w(a+b)) }
  - { model: modadd-1l, part: unembed, label: reads cos(w(a+b-c)) — constructive interference at the answer }
tags: [grokking, Fourier features, progress measures, weight decay, circles]
related: [number-helix, calendar-circles, geometric-calculator]
citations:
  - id: nanda2023
    title: Progress measures for grokking via mechanistic interpretability
    authors: Nanda, Chan, Lieberum, Smith, Steinhardt
    year: 2023
    venue: ICLR 2023
    url: https://arxiv.org/abs/2301.05217
  - id: power2022
    title: "Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets"
    authors: Power, Burda, Edwards, Babuschkin, Misra
    year: 2022
    venue: arXiv
    url: https://arxiv.org/abs/2201.02177
  - id: grokviz
    title: "Grokking, Visualized: modular addition, step by step (website, code, 22 runs)"
    authors: Burian
    year: 2026
    url: https://github.com/BurnyCoder/visualizing-grokking-modular-addition-in-transformer
  - id: nanda-repo
    title: progress-measures-paper (official code)
    authors: Nanda et al.
    year: 2023
    url: https://github.com/mechanistic-interpretability-grokking/progress-measures-paper
replications:
  - who: This atlas
    url: https://github.com/BurnyCoder/circuits-and-manifolds-from-mechanistic-interpretability-in-llms-and-other-models-atlas/blob/main/src/atlas/entries/grokking.py
    model: 1-layer transformer, p = 113, 3 seeds + no-weight-decay control
    outcome: ours
    note: Trained from scratch with the paper's hyper-parameters; Fourier analysis, neuron periodicity and a causal frequency ablation.
  - who: "Grokking, Visualized — 22 preregistered runs (Burian 2026)"
    url: https://burnycoder.github.io/visualizing-grokking-modular-addition-in-transformer/
    model: same 1-layer transformer; 11 seeds + no-weight-decay, random-label and fixed-split controls (Hugging Face BurnyCoder/grokking-modular-addition-runs)
    outcome: partial
    note: "Preregistered: 4 of 5 tests supported — grokking needs weight decay, restricted loss falls before test loss, excluded loss rises, Fourier sparsity is learned (11/11 seeds each); the strict key-frequency ablation test is not supported (6/11). Seed 0 starts from the paper's initial weights and ends with the paper's key frequencies. Includes a 22-chapter explorable explanation and a scrubbable dashboard."
---

## In plain words

Train a small network on a clock-arithmetic table — "what is (a + b) mod 113?" — showing it only 30% of the answers.
First it simply memorizes the training examples: training accuracy hits 100% while test accuracy stays near chance.
Thousands of steps later, with nothing changed, test accuracy suddenly jumps to 100%. This delayed generalization is
called **grokking** (Power et al. 2022).

Nanda et al. opened the trained network and found a clean algorithm. Every number is stored as a point on a few
circles (as angles). To add two numbers, the network adds their angles using trigonometry identities, then checks
which answer's angle matches. Weight decay slowly erases the memorized lookup table and leaves only this compact
circuit, which is why the jump happens late.

## How it works (technical)

With $w_k = 2\pi k / p$ for a handful of **key frequencies** $k$, the embedding maps $a \mapsto (\cos w_k a, \sin w_k a)$.
Attention and the MLP compute

$$
\cos(w_k(a+b)) = \cos w_k a \cos w_k b - \sin w_k a \sin w_k b,\quad \sin(w_k(a+b)) = \sin w_k a \cos w_k b + \cos w_k a \sin w_k b,
$$

and the unembedding reads $\text{logit}(c) \propto \sum_k \cos(w_k(a+b-c))$, which is maximal exactly at
$c \equiv a+b \pmod p$ (constructive interference). The paper's seed uses $k \in \{14, 35, 41, 42, 52\}$; 84.6% of
MLP neurons are explained (> 85% of variance) by a single frequency, and ablating the key frequencies destroys
performance while ablating the others slightly *improves* loss.

**What we measured** (3 seeds, {{epochs}} full-batch epochs, AdamW lr $10^{-3}$, weight decay 1):

- Seed 0 reaches ≥ 99% train accuracy at epoch {{train99_epoch_s0}} but ≥ 95% test accuracy only at epoch
  {{test95_epoch_s0}}; the worst seed ends at {{final_test_acc_min|pct}} test accuracy.
- Key frequencies per seed: {{key_freqs_all}} — different seeds pick different circles, as expected; at most
  {{n_freq_90_max}} of the 56 frequencies carry 90% of the embedding's norm.
- {{frac_periodic_s0|pct}} of seed-0 neurons have one dominant frequency (> 85% of variance).
- Removing the key frequencies from the embedding drops test accuracy from {{abl_clean|pct}} to {{abl_key|pct}};
  removing the same number of random other frequencies leaves {{abl_rand|pct}}.
- Without weight decay the model ends at {{wd0_final_train_acc|pct}} train / {{wd0_final_test_acc|pct}} test accuracy.

**Across 22 companion runs** (the "Grokking, Visualized" project's models, same architecture and settings; tab 6):
{{extra.n_wd1_grokked}} of {{extra.n_wd1}} weight-decay seeds grok, with test accuracy reaching 95% a median
{{extra.delay_median}}× later than train accuracy reaches 99% (minimum {{extra.delay_min}}×). Each seed settles on
{{extra.nkey_min}}–{{extra.nkey_max}} key frequencies of its own, with a median {{extra.frac_neurons_median|pct}} of
neurons dominated by one frequency. None of the {{extra.n_wd0}} no-weight-decay runs exceeds
{{extra.wd0_max_test|pct}} test accuracy, and the {{extra.n_rand}} random-label runs stay at chance
({{extra.rand_max_test|pct}} at best). For a 22-chapter, step-by-step explanation of this network, see
[Grokking, Visualized](https://burnycoder.github.io/visualizing-grokking-modular-addition-in-transformer/).

**Progress measures and the three phases** (tab 7; per-checkpoint metrics of the {{extra.n_prog}} weight-decay seeds).
Nanda et al. split training into *memorization*, *circuit formation* and *cleanup*. They show that two hidden quantities
move during the long stretch where train and test loss look flat. The *restricted loss*, which keeps only the key
frequencies of the logits, falls. The *excluded loss*, which removes them, rises on the training data. Re-implementing the
companion project's preregistered definitions in our own code, the restricted loss reaches 10% of its value at the end of
memorization {{extra.h1_lead_min}}–{{extra.h1_lead_max}} steps before the test loss does, in {{extra.n_h1}} of
{{extra.n_prog}} seeds. The excluded loss grows {{extra.h2_ratio_min}}× to {{extra.h2_ratio_max|sci}}× during circuit
formation, in {{extra.n_h2}} of {{extra.n_prog}} seeds. Our per-seed step counts match the companion's exactly in
{{extra.agree_companion}} of {{extra.n_prog}} seeds. The companion's other preregistered tests:
- Fourier sparsity is learned and specific to the Fourier basis: {{extra.comp_h3}} seeds.
- The *strict* key-frequency test passes in only {{extra.comp_h4}} seeds (supported: {{extra.comp_h4_supported}}).
  It requires every key frequency to cost ≥ 100× when ablated and the others jointly ≤ 1.5×. In some seeds the
  weight-norm rule that names the key frequencies admits a weak one or leaves out useful ones.

## Limitations

- One task, one architecture, one modulus; key frequencies are seed-dependent, so we compare counts and structure, not
  the exact set.
- Our "periodic neuron" fraction uses a simple 2-D Fourier variance criterion (one frequency's 3×3 block), similar but
  not identical to the paper's procedure.
- The progress-measure check re-implements the companion's definitions on its saved metrics. It is a code-level
  cross-check of a result we had already seen, not a blind replication, and not a recomputation from our own 3 seeds.
