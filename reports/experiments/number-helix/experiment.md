# Experiment: The number helix (trigonometric addition)

Code [`src/atlas/entries/number_helix.py`](../../../src/atlas/entries/number_helix.py) (the docstring gives the full method) · site
data `site/public/data/number-helix.json` · page `#/entry/number-helix` · pre-stated criteria and dated amendments:
[`reports/planned-runs.md`](../../planned-runs.md).

## Question

Do small Pythia models represent numbers on a generalized helix (Kantamneni & Tegmark 2025, [arXiv 2502.00873](https://arxiv.org/abs/2502.00873)), and does patching along the helix recover most of the causal effect of a layer?

## Setup

Five Pythia models (160M–2.8B). For each, the Fourier spectrum of number-token embeddings; held-out (80/20 over a) helix fits per layer with periods T = 2, 5, 10, 100, compared with a shuffled-a control and simpler bases; and activation patching (clean a+b vs corrupted a′+b, a+b < 100) for the full layer, the helix, circles only, linear only and PCA-9. Patching requires at least 20 correctly solved prompts (amendment).

## Results vs published claims

<!-- table:replication -->
| Claim | Published | Ours | Pre-stated criterion | Verdict |
|---|---|---|---|---|
| Number embeddings are periodic with periods 2, 5, 10 | Fourier peaks at T = 2, 5, 10 in GPT-J (Kantamneni & Tegmark Fig. 2); 2, 2.5, 5, 10 (Zhou et al. 2024) | pythia-2.8b: strongest embedding periods 100, 50, 33.3, 2 | at least two of {2, 5, 10} among the top-4 periods (±5%) | ❌ fail |
| Numbers lie on a generalized helix (T = 2, 5, 10, 100) | helix fits the residual stream of GPT-J / Pythia-6.9B / Llama-3.1-8B | pythia-2.8b layer 11: held-out R² 0.49 vs -0.09 for a shuffled helix, 0.24 linear-only | held-out helix R² ≥ 0.5 and ≥ 0.2 above the shuffled control | 🟡 partial |
| Patching along the helix recovers most of the full-layer effect | GPT-J: helix 7.21 vs full 8.34 logit diff (86%); PCA 6.13 | pythia-2.8b layer 1: median recovery helix 77%, full layer 100%, PCA-9 81%, linear-only 5% (n = 80 correct prompts; accuracy 78%) | helix ≥ 50% of the full-layer recovery and ≥ linear-only (needs ≥ 20 correctly solved prompts, else n/a) | ✅ pass |
<!-- /table:replication -->

## Interpretation

This is the first small-model evidence we know of. Pythia-1.4B and 2.8B solve two-digit addition well enough to test, and in 2.8B patching only the 9-dimensional helix subspace recovers most of the full-layer effect. The linear-only basis recovers almost nothing, which supports the trigonometric (not magnitude) account. The held-out helix fit is far above the shuffled control but lands just under the pre-stated R² threshold (partial). The embedding spectrum is dominated by long periods (100, 50, 33) and does not show the 2/5/10 peaks reported for larger models (fail).

## Limitations

- Models below 1B cannot add, so the causal test applies only to 1.4B and 2.8B.
- The paper's models (GPT-J, Pythia-6.9B, Llama-3.1-8B) are in the RunPod tier, not run here.

## All summary values

<!-- table:summary -->
| Key | Value |
|---|---|
| `largest` | pythia-2.8b |
| `best_layer` | 11 |
| `helix_test_r2` | 0.491567 |
| `shuffled_test_r2` | -0.091916 |
| `linear_test_r2` | 0.24157 |
| `circles_test_r2` | 0.4124 |
| `acc_largest` | 0.781726 |
| `patch_layer` | 1 |
| `patch_helix` | 0.769154 |
| `patch_full` | 0.99975 |
| `patch_pca9` | 0.810822 |
| `patch_linear` | 0.046502 |
| `patch_n` | 80 |
| `top_periods` | 100, 50, 33.3, 2 |
| `n_models` | 5 |
<!-- /table:summary -->
