---
slug: copy-suppression
title: Copy suppression (GPT-2 head 10.7)
category: circuit
collection: other
models: [GPT-2 small]
fidelity: recomputed
evidence: causal
order: 25
summary: A late attention head looks back at words the model is about to repeat and pushes them down — a built-in "don't just copy" brake.
viz: copy-suppression
data: copy-suppression.json
locations:
  - { model: gpt2-small, part: head, layer: 10, head: 7, label: copy-suppression head }
  - { model: gpt2-small, part: head, layer: 11, head: 10, label: second copy-suppression-like negative head }
tags: [negative heads, self-repair, calibration, OV circuit, QK circuit]
related: [ioi, induction-heads]
citations:
  - id: mcdougall2023
    title: "Copy Suppression: Comprehensively Understanding an Attention Head"
    authors: McDougall, Conmy, Rushing, McGrath, Nanda
    year: 2023
    venue: BlackboxNLP 2024
    url: https://arxiv.org/abs/2310.04625
  - id: wang2022
    title: "Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 small"
    authors: Wang, Variengien, Conmy, Shlegeris, Steinhardt
    year: 2022
    venue: ICLR 2023
    url: https://arxiv.org/abs/2211.00593
  - id: cs-app
    title: Copy suppression interactive explorer (Streamlit)
    authors: McDougall et al.
    year: 2023
    url: https://copy-suppression.streamlit.app/
replications:
  - who: This atlas
    url: https://github.com/BurnyCoder/circuits-and-manifolds-in-llms-atlas/blob/main/src/atlas/entries/copy_suppression.py
    model: GPT-2 small
    outcome: ours
    note: OV and QK weight analysis on 5,000 tokens, every other layer-10/11 head as a control, and eight worked prompts with zero-ablation.
---

## In plain words

Language models love to repeat words that already appeared. Often that is right ("Harry … Harry Potter"), but
sometimes it is a trap: after "All's fair in love and", the right continuation is "war", not "love" again. Head 10.7
in GPT-2 small is a specialist brake. When earlier layers start predicting a word that is already in the context, this
head attends back to that earlier copy and **lowers** the word's probability. In the IOI circuit the same head shows up
as a "negative name mover"; copy suppression is its general job.

## How it works (technical)

McDougall et al. describe three steps: (1) earlier layers raise the logit of a token that appears earlier in the
context; (2) L10H7's QK circuit, whose query reads "what is being predicted", attends to that earlier occurrence; (3)
its OV circuit writes the *negative* of that token's unembedding direction. They verify this from the weights, using
the "effective embedding" (the token embedding plus MLP0, since GPT-2's first MLP acts like an extended embedding):
84.70% of tokens have their own logit among the 10 most negative entries of the OV circuit, and 95.72% of rows of the
QK circuit peak on the diagonal. A copy-suppression-only ablation (CSPA) keeps 76.9% of the head's effect.

**What we measured** ({{n_tokens}} tokens, our row-wise ranking):

- OV: for {{ov_top10|pct}} of tokens, attending to a token puts that same token among the 10 most suppressed of 50,257
  outputs ({{ov_bottom5|pct}} land in the bottom 5%). The median other head in layers 10–11 scores {{ctrl_top10_median|pct}};
  the strongest other head is {{ctrl_top10_argmax}} ({{ctrl_top10_max|pct}}), IOI's second negative name mover.
- QK: when the query is a token's unembedding, the head attends most to that same token's effective embedding in
  {{qk_diag|pct}} of cases.
- Behaviour: on {{n_neg_direct}} of {{n_prompts}} hand-written prompts the head's direct effect on the repeated token's
  logit is negative, and zero-ablating it raises that token's probability on {{n_ablation_up}} of them. Mean attention
  to the earlier copy is {{mean_attn_copy}}.

## Limitations

- We rank within rows (effect of one source token on all outputs); the paper ranks within columns, so our percentages
  are comparable in spirit but not numerically identical.
- The effective embedding approximates layer-0 attention as attending to the token itself.
- Our prompt set is small and hand-written; the paper's behavioural statistics use OpenWebText.
