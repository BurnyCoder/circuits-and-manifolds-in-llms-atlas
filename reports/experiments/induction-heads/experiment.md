# Experiment: induction heads

Code [`src/atlas/entries/induction_heads.py`](../../../src/atlas/entries/induction_heads.py) · site data
`site/public/data/induction-heads.json` + `gpt2-map.json` · page `#/entry/induction-heads` · fidelity 🟢 recomputed.

## Question

Do the induction-head claims of Elhage et al. 2021 ([framework](https://transformer-circuits.pub/2021/framework/index.html))
and Olsson et al. 2022 ([in-context learning and induction heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html))
hold when recomputed from public checkpoints: (1) a 2-layer attention-only model has a previous-token head feeding
induction heads, (2) they causally implement copying of repeated sequences, (3) they form abruptly early in training?

## Setup

| Part | Model | Data | Measurement |
|---|---|---|---|
| Scores + ablation | TransformerLens `attn-only-2l` (2 layers × 8 heads, no MLPs) | 64 sequences of 25 random tokens repeated twice (seeded) | induction score = mean attention from position *i* in the second repeat to position *i−24* (the token after the previous occurrence); prev-token score = attention to *i−1* |
| Causal test | same | same | loss on the second repeat with heads zero-ablated (`hook_z` = 0) vs the same number of random non-induction layer-1 heads |
| GPT-2 map | GPT-2 small | same + 2 natural prompts | per-head induction / prev-token / duplicate-token scores and full attention patterns for the Model Explorer |
| Phase change | Pythia-160M, 18 checkpoints from step 0 to 143,000 (`PYTHIA_STEPS`) | repeated random tokens; 128 natural-text contexts | max induction score per checkpoint; ICL score = mean loss in tokens 494–504 minus tokens 44–54 (Olsson et al. use token 500 − 50) |

**Controls.** Random-init model with the same architecture (score baseline); random head ablation (same count) for the
causal test; non-repeated tokens implicitly via the first repeat.

**Why these settings.** Short repeated random-token sequences follow ARENA's induction chapter / TransformerLens demo;
64 sequences give bootstrap CIs on loss of about ±0.15 nats. Pythia checkpoints are log-spaced early because Olsson et al.
place the phase change at 2.5–5B tokens; Pythia logs 2,097,152 tokens per step, so step 1,000 ≈ 2.1B tokens.

## Results vs published claims

<!-- table:replication -->
| Claim | Published | Ours | Criterion | Verdict |
|---|---|---|---|---|
| A 2-layer attention-only model forms induction heads in layer 1 | Induction heads need ≥2 layers (Elhage et al. 2021); score defined by Olsson et al. 2022 | max layer-1 induction score 0.65 vs 0.031 at random init | trained max score > 0.3 and > 10× the random-init score | ✅ pass |
| Induction heads causally implement in-context copying | Ablating induction heads removes in-context learning in small attn-only models (Olsson et al., Argument 3) | repeat loss 2.68 [2.55, 2.83] → 12.45 [12.33, 12.56] nats when ablated (random-head control 2.64 [2.51, 2.79]) | ablation raises repeat loss by > 1 nat and > 3× the random-head control's increase | ✅ pass |
| Induction requires a previous-token head (K-composition) | Induction head keys read a previous-token head's output (Elhage et al. 2021) | ablating prev-token head 0.3 raises repeat loss to 11.22 nats | ablation raises repeat loss by > 1 nat | ✅ pass |
| Induction heads form abruptly early in training (phase change) | ICL score jumps in a narrow window at ~2.5–5B tokens in Anthropic's models (Olsson et al., Argument 1) | Pythia-160M: largest jump in max induction score between 1.07B and 2.10B tokens (0.03 at init → 0.96 final) | largest single-interval jump occurs before 20B tokens and accounts for > 50% of the total rise | ✅ pass |
<!-- /table:replication -->

## Interpretation

All four claims replicate. The causal test is the strongest evidence: zero-ablating the layer-1 induction heads pushes
loss on the repeated half back to roughly the loss on random tokens, while removing random heads changes almost
nothing. Ablating the previous-token head also breaks induction, the K-composition dependence described by Elhage et al.
In Pythia-160M the induction score jumps in a single early checkpoint interval, consistent with Olsson et al.'s
phase-change argument, though our checkpoint grid can only bound when the jump happens.

## Deviations from the paper

- ICL score uses 11-token windows and 128 contexts instead of single positions (noise; see learnings).
- Zero-ablation instead of mean-ablation for the causal test (attn-only models have no MLP to compensate, and the
  random-head control uses the same ablation).

## Limitations

- Correlation between induction-score jump and ICL-score jump is shown on one model size; Olsson et al. use many.
- The Pythia checkpoint grid is coarse (factor ~2 between neighbouring steps early on).

## All summary values

<!-- table:summary -->
| Key | Value |
|---|---|
| `a2_prev_head` | 0.3 |
| `a2_prev_score` | 0.519901 |
| `a2_ind_heads` | 1.6 |
| `a2_ind_max` | 0.649771 |
| `a2_rand_ind_max` | 0.031402 |
| `a2_loss_first` | 14.503569 |
| `a2_loss_second` | 2.67987 |
| `a2_loss_abl_ind` | 12.449474 |
| `a2_loss_abl_prev` | 11.221839 |
| `a2_loss_abl_rand` | 2.638943 |
| `gpt2_n_ind_heads` | 16 |
| `gpt2_top_ind` | 5.5 |
| `gpt2_top_ind_score` | 0.924532 |
| `pythia_jump_lo_B` | 1.073742 |
| `pythia_jump_hi_B` | 2.097152 |
| `pythia_final_ind` | 0.959 |
| `pythia_init_ind` | 0.030225 |
<!-- /table:summary -->
