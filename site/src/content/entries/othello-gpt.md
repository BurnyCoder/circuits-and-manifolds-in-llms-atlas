---
slug: othello-gpt
title: Othello-GPT's world model (mine vs. theirs)
category: manifold
collection: other
models: [Othello-GPT]
fidelity: recomputed
evidence: mixed
order: 40
summary: A transformer trained only on Othello move lists builds an internal, editable map of the board — linear once you say "mine" and "theirs" instead of black and white.
viz: othello
data: othello-gpt.json
locations:
  - { model: othello-gpt, part: resid, layers: [3, 7], label: linear mine/theirs/empty board representation (≈97% probe accuracy) }
  - { model: othello-gpt, part: resid, layers: [3, 5], label: editing the probe direction (best of these layers) changes predicted legal moves }
  - { model: othello-gpt, part: unembed, label: outputs a distribution over legal next moves }
tags: [world models, linear representation, probing, intervention]
related: [linebreak-manifold, calendar-circles]
citations:
  - id: li2023
    title: "Emergent World Representations: Exploring a Sequence Model Trained on a Synthetic Task"
    authors: Li, Hopkins, Bau, Viégas, Pfister, Wattenberg
    year: 2023
    venue: ICLR 2023
    url: https://arxiv.org/abs/2210.13382
  - id: li-gradient
    title: Do Large Language Models learn world models or just surface statistics?
    authors: Kenneth Li
    year: 2023
    venue: The Gradient
    url: https://thegradient.pub/othello/
  - id: nanda2023othello
    title: Emergent Linear Representations in World Models of Self-Supervised Sequence Models
    authors: Nanda, Lee, Wattenberg
    year: 2023
    venue: BlackboxNLP 2023
    url: https://arxiv.org/abs/2309.00941
  - id: othello-world
    title: othello_world (code, synthetic dataset, checkpoints)
    authors: Li et al.
    year: 2023
    url: https://github.com/likenneth/othello_world
replications:
  - who: This atlas
    url: https://github.com/BurnyCoder/circuits-and-manifolds-from-mechanistic-interpretability-in-llms-and-other-models-atlas/blob/main/src/atlas/entries/othello_gpt.py
    model: Othello-GPT (TransformerLens checkpoint)
    outcome: ours
    note: Legal-move rate, linear probes at every layer with three controls, and probe-direction interventions on new synthetic games from our own engine.
  - who: Nanda, Lee & Wattenberg — linear (not just nonlinear) board representation
    url: https://arxiv.org/abs/2309.00941
    model: Othello-GPT
    outcome: replicated
    note: Replaced Li et al.'s nonlinear MLP probes with linear mine/theirs probes (≥ 99% from layer 4; peak 99.6%) and showed simple vector-addition interventions work as well as gradient-based edits.
---

## In plain words

Othello-GPT never sees a board. It reads move sequences ("C4, E3, F6, …") and learns to predict a legal next move.
To do that well it seems to rebuild the board internally: a simple linear readout of its activations recovers which
squares are occupied and by whom. The twist found by Nanda et al.: the model doesn't think in "black" and "white" but in
**"mine" and "theirs"** — relative to whose turn it is. With that labeling the board becomes linearly readable, and you
can even *edit* it: push one square's activation from "theirs" to "mine" and the model changes which moves it
considers legal, as if the board had changed.

## How it works (technical)

For each layer $\ell$ we fit a linear probe $p_\ell: \mathbb{R}^{512} \to \{\text{empty}, \text{mine}, \text{theirs}\}^{64}$
on the residual stream after each move (cross-entropy, AdamW) using {{n_games_train}} synthetic random games, and test
on {{n_games_test}} held-out games. Controls: an absolute black/white probe on the same activations, the same
architecture at random initialization, and shuffled labels. For the intervention, take the probe weights $w_{s,c}$ for
square $s$, the direction $d = \widehat{w_{s,\text{target}} - w_{s,\text{current}}}$, and mirror the residual's component
along it at one layer: $x \leftarrow x - 2\,(x\cdot d)\,d$. We then compare the model's top-$k$ predicted moves with
the legal moves of the edited board (Li et al.'s error count).

**What we measured:**

- The model's top-1 move is illegal only {{illegal_rate|pct}} of the time.
- Mine/theirs probes reach {{acc_rel_best|pct}} accuracy (best layer {{best_layer}}); on occupied squares only
  {{occ_acc_best|pct}}. Absolute black/white probes top out at {{acc_abs_best|pct}}, probes on a random-init network at
  {{acc_rand_best|pct}}, and shuffled labels at {{acc_shuf_best|pct}} (≈ always guessing "empty").
- Flipping a square at layer {{int_layer}} (n = {{n_int}} edits that change the legal set): mean error against the edited
  board {{err_int}}, versus {{err_null}} with no intervention and {{err_rand}} for a random direction of the same size.

## Limitations

- Probes are correlational; the intervention is causal but uses a single layer and a single square.
- Our labels are relative to the player who just moved (equivalent to the paper's up to swapping the two classes).
- Games come from our own engine with the same rules as the original synthetic dataset; the model was trained on
  theirs.
