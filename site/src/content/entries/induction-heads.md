---
slug: induction-heads
title: Induction heads
category: circuit
collection: anthropic-tc
models: [attn-only-2l, GPT-2 small, Pythia-160M]
fidelity: recomputed
evidence: mixed
order: 10
summary: Two attention heads team up to continue repeated patterns — [A][B] … [A] → [B] — and they appear in a sudden phase change early in training.
viz: induction
data: induction-heads.json
locations:
  - { model: attn-only-2l, part: head, layer: 0, head: 3, label: previous-token head (writes 'the token before me') }
  - { model: attn-only-2l, part: head, layer: 1, head: 6, label: induction head (K-composes with 0.3) }
  - { model: pythia-160m, part: head, layer: 4, head: 6, label: strongest induction head after training }
tags: [in-context learning, copying, phase change, K-composition]
related: [ioi, copy-suppression, greater-than]
citations:
  - id: olsson2022
    title: In-context Learning and Induction Heads
    authors: Olsson, Elhage, Nanda, Joseph, DasSarma, Henighan, Mann, Askell, Bai, Chen, Conerly, … Olah
    year: 2022
    venue: Transformer Circuits Thread
    url: https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html
  - id: elhage2021
    title: A Mathematical Framework for Transformer Circuits
    authors: Elhage, Nanda, Olsson, Henighan, Joseph, Mann, Askell, Bai, Chen, Conerly, … Olah
    year: 2021
    venue: Transformer Circuits Thread
    url: https://transformer-circuits.pub/2021/framework/index.html
  - id: biderman2023
    title: "Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling"
    authors: Biderman, Schoelkopf, Anthony, Bradley, O'Brien, Hallahan, … van der Wal
    year: 2023
    venue: ICML 2023
    url: https://arxiv.org/abs/2304.01373
  - id: nanda-tl
    title: TransformerLens (attn-only-2l and other toy models)
    authors: Nanda, Bloom, et al.
    year: 2022
    url: https://github.com/TransformerLensOrg/TransformerLens
replications:
  - who: This atlas
    url: https://github.com/BurnyCoder/circuits-and-manifolds-from-mechanistic-interpretability-in-llms-and-other-models-atlas/blob/main/src/atlas/entries/induction_heads.py
    model: attn-only-2l, GPT-2 small, Pythia-160M checkpoints
    outcome: ours
    note: Head scores, zero-ablation causal test with a random-head control, and the training-time phase change (table below).
  - who: Olsson et al. (Anthropic), 34 models
    url: https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html#model-analysis-table
    model: 1–40-layer transformers up to 13B
    outcome: replicated
    note: The original paper itself spans small attention-only and large models; evidence is causal for small models and correlational for large ones.
---

## In plain words

Language models are good at continuing patterns they have just seen. If a document mentions "Harry Potter" once, the
model is much more confident that "Potter" follows the next "Harry". An **induction head** is the attention head that
does this lookup: when it sees token **A**, it searches the context for an earlier **A**, looks at the token that came
right after it (**B**), and boosts **B** as the next prediction.

It cannot do this alone. A second head in an *earlier* layer — a **previous-token head** — first writes "the token
before me was A" into every position. The induction head then matches its current token against those notes. This
two-step trick is why induction needs at least two layers.

## How it works (technical)

Write the residual stream at position $i$ as $x_i$. A previous-token head in layer 0 attends from $i$ to $i-1$ and
writes $W_{OV}^{(0)} x_{i-1}$ into position $i$. In layer 1, the induction head's attention score from destination
$j$ to source $i$ is

$$
s_{ji} = x_j^\top W_{QK}^{(1)} \big(x_i + W_{OV}^{(0)} x_{i-1}\big),
$$

so the key at $i$ carries information about the *previous* token (this is **K-composition**, Elhage et al. 2021). When
the current token equals the token before $i$, the score is high, and the head's OV circuit copies the token at $i$ to
the output. Olsson et al. define induction heads behaviourally: on a random token sequence repeated twice, a head is an
induction head if it attends from each token in the repeat to the token *after* that token's first occurrence
(prefix matching) and increases that token's logit (copying).

**What we measured.** On repeated random tokens ({{a2_loss_first}} nats loss on the first half — worse than uniform
guessing over the 48k-token vocabulary, 10.8 nats, because a trained model confidently expects real text),
the 2-layer attention-only model predicts the repeat almost perfectly ({{a2_loss_second}} nats). Its previous-token
head is {{a2_prev_head}} (score {{a2_prev_score}}), and the induction head(s) {{a2_ind_heads}} reach a score of
{{a2_ind_max}}, versus {{a2_rand_ind_max}} for the same architecture at random initialization. Zero-ablating the
induction heads raises the repeat loss to {{a2_loss_abl_ind}} nats and ablating the single previous-token head raises
it to {{a2_loss_abl_prev}} nats, while ablating non-induction layer-1 heads instead gives {{a2_loss_abl_rand}}
nats on average over every same-size set (the review asked for all sets instead of one random draw). GPT-2 small has {{gpt2_n_ind_heads}} heads with an induction score above 0.3 (strongest: {{gpt2_top_ind}}, score
{{gpt2_top_ind_score}}).

**Phase change.** Across Pythia-160M checkpoints, the strongest induction score rises from {{pythia_init_ind}} at
initialization to {{pythia_final_ind}}, with the largest single jump between {{pythia_jump_lo_B}}B and
{{pythia_jump_hi_B}}B training tokens — early in a 300B-token run, matching the abrupt early formation Olsson et al.
report (their Argument 1).

## Limitations

- Scores are behavioural (attention patterns on synthetic repeated tokens); we do not check the OV copying circuit
  directly here (see the copy-suppression entry for an OV analysis).
- Pythia checkpoints are sparse after step 1,000 (every 1,000 steps ≈ 2.1B tokens), so the jump is localized only to
  within one checkpoint interval.
- In large models induction heads coexist with fuzzier variants (translation, pattern-matching heads); Olsson et al.'s
  evidence that they drive *most* in-context learning is correlational for large models.
