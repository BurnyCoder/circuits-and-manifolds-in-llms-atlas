---
slug: geometric-calculator
title: A geometric calculator (base-10 circles)
category: circuit
collection: goodfire-geometry
models: [Llama-3.1-8B (paper), Pythia-410M/1.4B/2.8B (our probes)]
fidelity: replicated
evidence: causal
order: 640
summary: Asked "16 months after August?", Llama turns August into 8, adds 16 on base-10 circles like a calculator, and maps 24 back to a month. Fourier probes in Pythia show only weak parity and mod-10 signals.
viz: geometric-calculator
data: calculator-probes.json
locations:
  - { model: llama-3.1-8b, part: mlp, layer: 18, label: 28 shared neurons compute the sum on circles of period 2, 5, 10 (Feucht et al.) }
  - { model: pythia-160m, part: resid, layers: [3, 9], label: Fourier probes for (a+b) on base-10 circles (our probes run on Pythia-410M–2.8B) }
tags: [arithmetic, Fourier features, circles, cyclic concepts]
related: [number-helix, calendar-circles, grokking, manifold-steering]
citations:
  - id: feucht2026
    title: "Arithmetic in the Wild: Llama uses Base-10 Addition to Reason About Cyclic Concepts"
    authors: Feucht, Haklay, Bhalla, Wurgaft, Rager, Sarfati, Merullo, McGrath, Lewis, Lubana, Fel, Geiger
    year: 2026
    venue: arXiv (CC BY 4.0)
    url: https://arxiv.org/abs/2605.01148
  - id: calc-blog
    title: A Geometric Calculator Inside a Neural Network (Neural Geometry series)
    authors: Goodfire
    year: 2026
    url: https://www.goodfire.com/research/a-geometric-calculator
  - id: kantamneni2025
    title: Language Models Use Trigonometry to Do Addition
    authors: Kantamneni, Tegmark
    year: 2025
    url: https://arxiv.org/abs/2502.00873
replications:
  - who: This atlas — Fourier probes in small models
    url: https://github.com/BurnyCoder/circuits-and-manifolds-from-mechanistic-interpretability-in-llms-and-other-models-atlas/blob/main/src/atlas/entries/calculator_probes.py
    model: Pythia-410M / 1.4B / 2.8B
    outcome: ours
    note: Same probe design (sin/cos of 2π(a+b)/T, T = 2…150) on small models; the paper reports no small-model evidence. Confirmatory test with a fixed layer, a leak-free pair split, an R² floor and two nulls.
---

## In plain words

How would you compute "16 months after August"? Feucht et al. found Llama-3.1-8B does something surprisingly
calculator-like. It converts August to the number 8, adds 16 using the same machinery it uses for ordinary addition —
numbers stored on little circles that repeat every 2, 5 and 10 (like the ones digit and odd/even) — and turns 24 back
into a month (December). A small set of 28 neurons in layer 18 does the adding for months, weekdays, hours and plain
numbers alike.

## How it works (technical)

Affine probes on the final token of "{a}+{b}=" recover $\sin/\cos(2\pi(a+b)/T)$ best for $T \in \{2, 5, 10, 20, 50, 100\}$;
distributed alignment search localizes the input, offset and output subspaces at layer 18; 28 MLP neurons (≈ 0.2% of the
layer) write to those circles. Zeroing them drops weekday accuracy from 91.8% to 40.8% and addition from 94.6% to 24.1%;
patching addition results into calendar prompts converts the calendar answers.

**What we measured** (Fourier probes, {{model}}, layer {{layer}}): the eight best-decoded periods are {{top_periods}},
{{n_base10_in_top8}} of them base-10 periods; $R^2$ at $T = 10$ is {{r2_T10}} vs {{r2_T10_neighbours}} at the
neighbouring periods 9 and 11 ($T = 5$: {{r2_T5}}, $T = 2$: {{r2_T2}}).

**Confirmatory test** (pre-stated 2026-11-28): layer fixed in advance at ⌊2/3⌋ depth (layer {{conf_layer}}), train and
test prompts split by unordered pair {a, b} so that "a+b" and "b+a" never straddle the split, and a period counts only
if its held-out $R^2$ exceeds 0.05. The top-8 such peaks are {{conf_valid_top}}; the base-10 hits are {{conf_hits}}
($R^2$ at $T = 5$ is {{conf_r2_T5}}, at $T = 10$ {{conf_r2_T10}}). Both nulls are passed (label permutation p =
{{conf_p_perm}}, random period sets p = {{conf_p_set}}), but two hits make the row partial. With the leak-free split,
$R^2(T = 10)$ drops well below the row-split value above: some of that signal came from repeated pairs. Smaller models:
{{conf_other_models}}. The evidence is for parity and a weak mod-10 code, not for a set of base-10 circles.

## Limitations

- The mechanism itself (neurons, DAS, cross-task patching) is from the paper on Llama-3.1-8B; we only probe small Pythia
  models for the base-10 circles. The clock in tab 1 is an illustration, not model data.
