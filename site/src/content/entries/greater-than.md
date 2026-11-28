---
slug: greater-than
title: Greater-than circuit
category: circuit
collection: other
models: [GPT-2 small]
fidelity: recomputed
evidence: causal
order: 27
summary: '"The war lasted from 1732 to 17…" — attention heads move the start year to the end, and MLPs 8–11 turn it into "any year after 32".'
viz: greater-than
data: greater-than.json
locations:
  - { model: gpt2-small, part: head, layer: 5, head: 1, label: greater-than year mover }
  - { model: gpt2-small, part: head, layer: 5, head: 5, label: greater-than year mover }
  - { model: gpt2-small, part: head, layer: 6, head: 9, label: greater-than year mover }
  - { model: gpt2-small, part: head, layer: 7, head: 10, label: greater-than year mover }
  - { model: gpt2-small, part: head, layer: 8, head: 8, label: greater-than year mover }
  - { model: gpt2-small, part: head, layer: 8, head: 11, label: greater-than year mover }
  - { model: gpt2-small, part: head, layer: 9, head: 1, label: greater-than year mover }
  - { model: gpt2-small, part: mlp, layer: 8, label: computes 'years after YY' }
  - { model: gpt2-small, part: mlp, layer: 9, label: computes 'years after YY' }
  - { model: gpt2-small, part: mlp, layer: 10, label: computes 'years after YY' }
  - { model: gpt2-small, part: mlp, layer: 11, label: computes 'years after YY' }
tags: [mathematical reasoning, MLPs, path patching]
related: [ioi, number-helix, grokking]
citations:
  - id: hanna2023
    title: "How does GPT-2 compute greater-than?: Interpreting mathematical abilities in a pre-trained language model"
    authors: Hanna, Liu, Variengien
    year: 2023
    venue: NeurIPS 2023
    url: https://arxiv.org/abs/2305.00586
  - id: hanna-code
    title: gpt2-greater-than (official code)
    authors: Hanna et al.
    year: 2023
    url: https://github.com/hannamw/gpt2-greater-than
  - id: merullo2025
    title: Replicating Circuit Tracing for a Simple Known Mechanism
    authors: Merullo et al. (Goodfire)
    year: 2025
    url: https://www.goodfire.ai/research/replicating-circuit-tracing-for-a-simple-mechanism
replications:
  - who: This atlas
    url: https://github.com/BurnyCoder/circuits-and-manifolds-from-mechanistic-interpretability-in-llms-and-other-models-atlas/blob/main/src/atlas/entries/greater_than.py
    model: GPT-2 small
    outcome: ours
    note: Probability difference, year × year heatmaps for all 32 on/off patching combinations, circuit-only faithfulness and a random-head control.
  - who: Goodfire — replicating circuit tracing for a simple mechanism
    url: https://www.goodfire.ai/research/replicating-circuit-tracing-for-a-simple-mechanism
    model: GPT-2 small + cross-layer transcoder
    outcome: partial
    note: Finds greater-than features plus parity and multiple-of-5 features, but the top features' logit attributions barely depend on the start year, unlike Hanna et al.'s MLP neurons; one feature does the inverse (promotes small years).
---

## In plain words

Give GPT-2 small "The war lasted from the year 1732 to the year 17" and it bets on years after 32. Hanna et al. found
how: a few attention heads at the final position look back at the start year's two digits ("32") and copy that
information forward; then MLPs in layers 8–11 turn "the start was 32" into "boost 33, 34, … 99 and suppress 00–32". In
the heatmap this is a sharp edge along the diagonal: probability mass sits to the right of the start year.

## How it works (technical)

The task metric is the probability difference
$\text{PD} = \sum_{y > YY} p(y) - \sum_{y \le YY} p(y)$ over the 100 two-digit continuations (paper: 81.7%). Path
patching from a corrupted prompt with start year "01" isolates heads 5.1, 5.5, 6.9, 7.10, 8.8, 8.11 and 9.1 (moving YY
to the END position) and MLPs 8–11, whose logit-lens outputs form an upper-triangular pattern; the circuit alone keeps
89.5% of PD.

**What we measured** ({{n_prompts}} prompts, {{n_years}} start years 1702–1798):

- Clean PD = {{pd_clean|pct}}; with the start year replaced by 01 it is {{pd_corrupt|pct}}.
- Patching only the seven circuit heads with their "01" activations gives PD {{pd_heads_patched|pct}}; patching only
  MLPs 8–11 gives {{pd_mlps_patched|pct}}; both together {{pd_all_patched|pct}}.
- Patching seven *random* heads leaves a median PD of {{rand_pd_median|pct}}.
- Circuit only (all 137 other heads patched, MLPs untouched): PD {{pd_circuit_only|pct}}, i.e. {{faithful_frac|pct}} of the
  full model.

## Limitations

- We use activation patching of whole components rather than the paper's edge-level path patching.
- Our circuit-only test leaves all MLPs clean (patching early MLPs at the year token deletes the year itself — see
  `reports/experiments/learnings.md`), so it measures the sufficiency of the seven heads given the MLPs, not of the
  full head+MLP circuit in isolation.
- Only the 18th century and our noun list; the paper used several centuries.
