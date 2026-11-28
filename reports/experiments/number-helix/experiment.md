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
| Claim | Published | Ours | Criterion | Verdict |
|---|---|---|---|---|
| Number embeddings are periodic with periods 2, 5, 10 | Fourier peaks at T = 2, 5, 10 in GPT-J (Kantamneni & Tegmark Fig. 2); 2, 2.5, 5, 10 (Zhou et al. 2024) | pythia-2.8b: strongest embedding periods 100, 50, 33.3, 2 | at least two of {2, 5, 10} among the top-4 periods (±5%) | ❌ fail |
| Numbers lie on a generalized helix (T = 2, 5, 10, 100) | helix fits the residual stream of GPT-J / Pythia-6.9B / Llama-3.1-8B | pythia-2.8b: 5-fold CV R² over a, max over layers for each basis: helix 0.50 (layer 11) vs -0.11 for a shuffled helix, 0.26 linear-only, 0.38 circles only | held-out helix R² ≥ 0.5 and ≥ 0.2 above the shuffled control (control fix 2026-11-28: CV, controls max-selected too) | 🟡 partial |
| Patching along the helix recovers most of the full-layer effect | GPT-J: helix 7.21 vs full 8.34 logit diff (86%); PCA 6.13 | pythia-2.8b layer 1: median recovery helix 76%, full layer 100%, PCA-9 80%, linear-only 4% (n = 80 correct prompts; accuracy 85%) | helix ≥ 50% of the full-layer recovery and ≥ linear-only; full-layer recovery ≥ 50% (m7 guard, 2026-11-28) (needs ≥ 20 correctly solved prompts, else n/a) | ✅ pass |
| Confirmatory: patching along the helix beats generic 9-D subspaces | GPT-J: helix 7.21 vs PCA 6.13 logit difference (Kantamneni & Tegmark Table 1) | pythia-2.8b, mean per-prompt recovery, paired differences [95% bootstrap CI]: L1: helix 72% vs random 9-D 2% (diff +70 [+66, +73]) vs PCA-9 78% (diff -6 [-9, -2]); L8: helix 69% vs random 9-D 4% (diff +65 [+62, +68]) vs PCA-9 65% (diff +4 [-1, +9]) | CI > 0 against random 9-D and PCA-9 at every layer with full-layer recovery ≥ 50%; partial if only against random 9-D (pre-stated 2026-11-28) | 🟡 partial |
<!-- /table:replication -->

## Interpretation

Pythia-1.4B and 2.8B solve two-digit addition well enough to test, and in 2.8B patching the 9-dimensional helix
subspace recovers most of the full-layer effect at early layers (the pre-stated criterion passes). **This is not
helix-specific:** a dimension-matched 9-D PCA subspace recovers about as much (`patch_pca9` vs `patch_helix` at layer `patch_layer`
in the summary table below; slightly more, in fact), whereas the paper finds the helix ahead of PCA (7.21 vs 6.13 logit
difference, Kantamneni & Tegmark Table 1). The 1-D linear patch recovering almost nothing is not evidence for the
trigonometric account, because it has 8 fewer dimensions. By mid-depth the full-layer recovery itself is near zero, so
the information has left the operand token. The held-out helix fit is far above the shuffled control but lands just
under the pre-stated R² threshold (partial). The embedding spectrum is dominated by long periods (100, 50, 33) and
does not show the 2/5/10 peaks reported for larger models (fail). Details: [methods review](../../reviews/methods-review.md), M7.

**Control fix and confirmatory test (pre-stated 2026-11-28).**
- *Row 2* now uses 5-fold CV over a and takes the maximum over layers for every basis, the shuffled helix included
  (`helix_cv_r2` at `cv_layer` vs `shuffled_cv_r2`). The helix is still far above the shuffled control and still just
  under 0.5, so the verdict stays partial.
- *Row 3* now also requires the full-layer recovery to be ≥ 0.5 (m7); it is, so the row still passes.
- *Confirmatory patch test: partial.* At every patch layer where the full layer recovers ≥ 50% (layers 1 and 8 in
  Pythia-2.8B), paired per-prompt differences (`patch_spec`): the helix beats a random 9-D subspace of the 99-D
  number-activation span by a wide margin at both layers, but it does not beat PCA-9. PCA-9 is ahead at layer 1 and
  tied within the CI at layer 8. Unlike GPT-J in the paper, Pythia-2.8B gives no evidence that the helix basis is
  better than the top principal directions of the same activations.

## Limitations

- Models below 1B cannot add, so the causal test applies only to 1.4B and 2.8B.
- The original single-split helix numbers (`helix_test_r2`) are kept on the site for continuity; the verdict uses CV.
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
| `acc_largest` | 0.846939 |
| `patch_layer` | 1 |
| `patch_helix` | 0.758131 |
| `patch_full` | 1.000409 |
| `patch_pca9` | 0.798555 |
| `patch_linear` | 0.041833 |
| `patch_n` | 80 |
| `top_periods` | 100, 50, 33.3, 2 |
| `n_models` | 5 |
| `helix_cv_r2` | 0.499817 |
| `shuffled_cv_r2` | -0.106889 |
| `linear_cv_r2` | 0.259422 |
| `circles_cv_r2` | 0.378872 |
| `cv_layer` | 11 |
| `patch_spec` | L1: helix 72% vs random 9-D 2% (diff +70 [+66, +73]) vs PCA-9 78% (diff -6 [-9, -2]); L8: helix 69% vs random 9-D 4% (diff +65 [+62, +68]) vs PCA-9 65% (diff +4 [-1, +9]) |
<!-- /table:summary -->
