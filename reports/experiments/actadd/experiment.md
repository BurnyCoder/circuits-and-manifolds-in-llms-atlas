# Experiment: Activation addition (ActAdd)

Code [`src/atlas/entries/actadd.py`](../../../src/atlas/entries/actadd.py) (the docstring gives the full method) · site
data `site/public/data/actadd.json` · page `#/entry/actadd` · pre-stated criteria and dated amendments:
[`reports/planned-runs.md`](../../planned-runs.md).

## Question

Can adding the activation difference of a contrast prompt pair steer GPT-2 XL's topic and sentiment (Turner et al. 2023, [arXiv 2308.10248](https://arxiv.org/abs/2308.10248))?

## Setup

GPT-2 XL in bfloat16; steering vectors from "I talk about weddings constantly" − "I do not …" (layer 20) and "Love" − "Hate" (layer 6), added to the first prompt positions with coefficients 0–8; 8 samples × 3 prompts per setting, the paper's sampling settings; norm-matched random-vector control; topic adherence by keyword, sentiment by a DistilBERT SST-2 classifier.

## Results vs published claims

<!-- table:replication -->
| Claim | Published | Ours | Pre-stated criterion | Verdict |
|---|---|---|---|---|
| Adding a 'weddings' vector steers the topic | > 90% of completions about weddings vs ~2% unsteered (Turner et al.) | wedding words in 0% of completions unsteered → 75% at c = 4 (random vector 0%); best 92% | c = 4 rate ≥ 50% and ≥ 30 points above both the unsteered and random-vector rates | ✅ pass |
| A 'Love − Hate' vector makes text more positive | Love − Hate at layer 6 turns 'I hate you because' positive (Turner et al.) | positive sentiment 54% → 42% at c = 4 (random vector 42%) | ≥ 20 points above both baseline and random vector | ❌ fail |
<!-- /table:replication -->

## Interpretation

The weddings vector replicates. On-topic completions rise steeply with the coefficient to the paper's level, while the random vector does nothing. The Love − Hate vector does not shift classifier-measured sentiment at any coefficient in our setup (fail). The paper's evidence for it was largely qualitative examples.

## Limitations

- Sentiment is measured on prompt + completion; negative prompts ("I hate you because") anchor the classifier.
- 24 completions per setting.

## All summary values

<!-- table:summary -->
| Key | Value |
|---|---|
| `wedding_base` | 0.0 |
| `wedding_c4` | 0.75 |
| `wedding_rand_c4` | 0.0 |
| `wedding_max` | 0.916667 |
| `love_base` | 0.541667 |
| `love_c4` | 0.416667 |
| `love_rand_c4` | 0.416667 |
| `coef_max` | 8 |
| `n_per_prompt` | 8 |
<!-- /table:summary -->
