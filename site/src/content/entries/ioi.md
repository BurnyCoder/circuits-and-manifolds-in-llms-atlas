---
slug: ioi
title: Indirect object identification (IOI) circuit
category: circuit
collection: redwood
models: [GPT-2 small]
fidelity: recomputed
evidence: causal
order: 20
summary: 26 attention heads in 7 classes let GPT-2 small answer "When Mary and John went to the store, John gave a drink to" → " Mary".
viz: ioi
data: ioi.json
locations:
  - { model: gpt2-small, part: head, layer: 9, head: 9, label: IOI name mover }
  - { model: gpt2-small, part: head, layer: 9, head: 6, label: IOI name mover }
  - { model: gpt2-small, part: head, layer: 10, head: 0, label: IOI name mover }
  - { model: gpt2-small, part: head, layer: 10, head: 7, label: IOI negative name mover }
  - { model: gpt2-small, part: head, layer: 11, head: 10, label: IOI negative name mover }
  - { model: gpt2-small, part: head, layer: 7, head: 3, label: IOI S-inhibition }
  - { model: gpt2-small, part: head, layer: 7, head: 9, label: IOI S-inhibition }
  - { model: gpt2-small, part: head, layer: 8, head: 6, label: IOI S-inhibition }
  - { model: gpt2-small, part: head, layer: 8, head: 10, label: IOI S-inhibition }
  - { model: gpt2-small, part: head, layer: 5, head: 5, label: IOI induction }
  - { model: gpt2-small, part: head, layer: 6, head: 9, label: IOI induction }
  - { model: gpt2-small, part: head, layer: 5, head: 8, label: IOI induction }
  - { model: gpt2-small, part: head, layer: 5, head: 9, label: IOI induction }
  - { model: gpt2-small, part: head, layer: 0, head: 1, label: IOI duplicate token }
  - { model: gpt2-small, part: head, layer: 3, head: 0, label: IOI duplicate token }
  - { model: gpt2-small, part: head, layer: 0, head: 10, label: IOI duplicate token }
  - { model: gpt2-small, part: head, layer: 2, head: 2, label: IOI previous token }
  - { model: gpt2-small, part: head, layer: 4, head: 11, label: IOI previous token }
  - { model: gpt2-small, part: head, layer: 9, head: 0, label: IOI backup name mover }
  - { model: gpt2-small, part: head, layer: 9, head: 7, label: IOI backup name mover }
  - { model: gpt2-small, part: head, layer: 10, head: 1, label: IOI backup name mover }
  - { model: gpt2-small, part: head, layer: 10, head: 2, label: IOI backup name mover }
  - { model: gpt2-small, part: head, layer: 10, head: 6, label: IOI backup name mover }
  - { model: gpt2-small, part: head, layer: 10, head: 10, label: IOI backup name mover }
  - { model: gpt2-small, part: head, layer: 11, head: 2, label: IOI backup name mover }
  - { model: gpt2-small, part: head, layer: 11, head: 9, label: IOI backup name mover }
tags: [path patching, name movers, backup heads, self-repair]
related: [induction-heads, copy-suppression, greater-than]
quote:
  text: Our explanation encompasses 26 attention heads grouped into 7 main classes.
  source: wang2022
  where: Abstract
citations:
  - id: wang2022
    title: "Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 small"
    authors: Wang, Variengien, Conmy, Shlegeris, Steinhardt
    year: 2022
    venue: ICLR 2023
    url: https://arxiv.org/abs/2211.00593
  - id: radford2019
    title: Language Models are Unsupervised Multitask Learners (GPT-2)
    authors: Radford, Wu, Child, Luan, Amodei, Sutskever
    year: 2019
    venue: OpenAI
    url: https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
  - id: mcdougall2023
    title: "Copy Suppression: Comprehensively Understanding an Attention Head"
    authors: McDougall, Conmy, Rushing, McGrath, Nanda
    year: 2023
    venue: BlackboxNLP 2024
    url: https://arxiv.org/abs/2310.04625
replications:
  - who: This atlas
    url: https://github.com/BurnyCoder/circuits-and-manifolds-from-mechanistic-interpretability-in-llms-and-other-models-atlas/blob/main/src/atlas/entries/ioi.py
    model: GPT-2 small
    outcome: ours
    note: Logit difference, direct logit attribution, activation noising, all 128 class knockouts, faithfulness and a random-head control on our own templates.
  - who: McDougall et al. — copy suppression explains the negative name movers
    url: https://arxiv.org/abs/2310.04625
    model: GPT-2 small
    outcome: replicated
    note: Shows negative name mover 10.7 is a general copy-suppression head; it accounts for part of the backup/self-repair behavior IOI found.
---

## In plain words

Read "When Mary and John went to the store, John gave a drink to". The natural next word is **Mary**: John is already
mentioned twice and is the one *giving*. GPT-2 small gets this right almost every time. Wang et al. traced the
attention heads that make it happen, and the algorithm is surprisingly human-readable:

1. **Find the duplicate.** Some heads notice that "John" appears twice (duplicate-token and induction heads, helped by
   previous-token heads).
2. **Inhibit it.** *S-inhibition* heads carry the message "John is repeated — don't say John" to the final position.
3. **Copy the other name.** *Name mover* heads attend from the last token to the names, are told to ignore John, and
   copy **Mary** into the output.

There are also *negative* name movers that push against the answer (hedging), and *backup* name movers that take over
if the main ones are knocked out.

## How it works (technical)

The metric is the logit difference $\Delta = \text{logit}(\text{IO}) - \text{logit}(\text{S})$ at the final token. Wang
et al. find $\Delta = 3.56$ on their data with IO preferred in 99.3% of prompts, and identify heads with *path
patching* from a corrupted distribution $p_{\text{ABC}}$ where the three name slots hold three different names. Name
movers' OV circuits copy names (copy score > 95%) and their attention to IO correlates with their effect on the
logits; the circuit keeps 87% of $\Delta$ when everything else is mean-ablated.

**What we measured** (our templates, {{n_prompts}} prompts, GPT-2 small):

- Mean logit difference {{ld_clean}} (95% CI {{ld_clean_lo}}–{{ld_clean_hi}}); IO beats S in {{io_win|pct}} of
  prompts. On the corrupted $p_{\text{ABC}}$ prompts it falls to {{ld_abc}}.
- Name movers 9.9 / 9.6 / 10.0 put on average {{nm_attn_io}} of their final-token attention on the IO name versus
  {{nm_attn_s}} on the subject.
- Direct logit attribution of the negative name movers is {{neg_nm_dla_10_7}} (10.7) and {{neg_nm_dla_11_10}} (11.10):
  they write against the correct answer. The head with the largest positive direct effect is {{top_dla_head}}.
- Keeping only the 26 circuit heads (mean-ablating the other 118) leaves a logit difference of {{faithful_ld}}, i.e.
  {{faithful_frac|pct}} of the full model. Knocking out all seven classes drops it to {{all_classes_ablated_ld}}, while
  knocking out 26 random heads leaves a median of {{rand_ablated_ld_median}}.
- With the paper's node definition, (head, position) pairs (each class kept only at its position: previous-token heads
  at S1+1, duplicate-token and induction heads at S2, the rest at END), the circuit keeps {{faithful_pos_frac|pct}}
  (95% CI {{faithful_pos_frac_lo|pct}}–{{faithful_pos_frac_hi|pct}}). This confirmatory test was pre-stated on
  2026-11-28. Restricting positions does not lower faithfulness, so the gap to the paper's 87% is not explained by our
  keeping heads at every position.

## Limitations

- We use activation noising and direct logit attribution instead of the paper's full path patching, so the importance
  maps mix direct and indirect effects differently from the paper's Figures 3–5.
- Templates are ours (8 templates, single-token names/places/objects); absolute numbers differ from the paper's
  dataset.
- The circuit is specific to this narrow task; MLPs and LayerNorm were not studied in the original work.
- Backup heads mean "importance" depends on what else is ablated (self-repair); see the knockout explorer.
