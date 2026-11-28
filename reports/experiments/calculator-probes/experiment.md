# Experiment: The geometric calculator (Fourier probes)

Code [`src/atlas/entries/calculator_probes.py`](../../../src/atlas/entries/calculator_probes.py) (the docstring gives the full method) · site
data `site/public/data/calculator-probes.json` · page `#/entry/geometric-calculator` · pre-stated criteria and dated amendments:
[`reports/planned-runs.md`](../../planned-runs.md).

## Question

Do small Pythia models encode addition results on base-10 Fourier circles (periods 2, 5, 10, …) like Llama-3.1-8B (Feucht et al. 2026, [arXiv 2605.01148](https://arxiv.org/abs/2605.01148))?

## Setup

Last-token activations of "{a}+{b}=" (a, b ∈ 1…99) in Pythia-410M, 1.4B and 2.8B at several layers; ridge probes to sin/cos(2π(a+b)/T) for T = 2…150 with α chosen per period on training rows; held-out R². Pre-stated: base-10 periods in the global top 8. Amendment: top 8 by peak score R²(T) − mean R²(T±1).

## Results vs published claims

<!-- table:replication -->
| Claim | Published | Ours | Criterion | Verdict |
|---|---|---|---|---|
| Addition results are encoded on base-10 circles (T = 2, 5, 10, …) | Fourier probes peak at T ∈ {2, 5, 10, 20, 50, 100} in Llama-3.1-8B (Feucht et al.) | pythia-2.8b layer 21: top-8 periods 150, 149, 148, 147, 146, 145, 144, 143 (0 of the base-10 set); R²(T=10) 0.20 vs neighbours -0.01 | ≥ 3 base-10 periods in the top 8 and R²(T=10) ≥ neighbours + 0.1 | ❌ fail |
| Base-10 periods stand out as peaks (amendment: peak score, removes the long-period magnitude trend) | Fourier probes peak at T ∈ {2, 5, 10, 20, 50, 100} in Llama-3.1-8B (Feucht et al.) | pythia-2.8b layer 21: top-8 periods by R²(T) − mean R²(T±1): 2, 10, 38, 58, 5, 7, 15, 17 (3 of the base-10 set) | ≥ 3 base-10 periods among the top-8 peaks and R²(T=10) ≥ neighbours + 0.1 (same thresholds) | ✅ pass (post-hoc: peak-score analysis added after the pre-stated ranking failed) |
| Confirmatory: base-10 periods stand out at a fixed layer with an R² floor and nulls | Fourier probes peak at T ∈ {2, 5, 10, 20, 50, 100} in Llama-3.1-8B (Feucht et al.) | pythia-2.8b layer 21 (⌊2n/3⌋), split by unordered pair: top-8 peaks with R² > 0.05: 2, 10, 71, 39, 150, 43, 44, 64; base-10 hits 2, 10 (R²(5) -0.01, R²(10) 0.09); label-permutation p 0.005, random-period-set p 0.0318. Other models: pythia-410m L16: hits none; pythia-1.4b L16: hits [2] | ≥ 3 hits with both p < 0.05; partial with 2 hits (pre-stated 2026-11-28) | 🟡 partial |
<!-- /table:replication -->

## Interpretation

Weak evidence for a mod-10 code, not for base-10 circles. The amended peak-score row is post-hoc (added after the
pre-stated ranking failed) and is excluded from the tally. Even on its own terms it is weak: one of its three base-10
"peaks", T = 5, has held-out R² near zero (see the per-period curves on the site), so the pass rests on T = 2 (parity) and T = 10 (R² about
0.2 in late layers). Non-base-10 periods (4, 7, 15) also recur among the peaks, the peak score has no null, the verdict
layer is the one that maximises R²(2) + R²(5) + R²(10), and only two of four layers give three hits. The pre-stated
global ranking fails because periods longer than the range of a+b act as a monotone magnitude code. Details: [methods review](../../reviews/methods-review.md), M6.

**Confirmatory test (pre-stated 2026-11-28): partial.** All of the review's fixes at once: layer fixed at ⌊2n/3⌋, split
by unordered pair {a, b}, a period counts only if R²(T) > 0.05, a label-permutation null (200 shuffles of the sums
through the same probes) and a random-period-set null. Only T = 2 and T = 10 survive as base-10 hits (`conf_hits`);
T = 5 has R² near zero (`conf_r2_T5`). Both nulls reject (`conf_p_perm`, `conf_p_set`), so the two hits are not
chance, but the criterion needs three. With the leak-free split, R²(T = 10) at the same layer (`conf_r2_T10`) is less
than half the row-split value (`r2_T10`): the row split let duplicate and commuted prompts inflate the probe. The two
smaller models have at most the parity hit (`conf_other_models`). Pythia up to 2.8B shows parity plus a weak mod-10
code; the base-10 circle family reported for Llama-3.1-8B is not visible at this scale.

## Limitations

- The paper's 28-neuron ablations and DAS are not reproduced; probes are correlational.
- The original rows use a row split, so some of their test prompts repeat training prompts or their commuted form
  (b + a); the confirmatory row fixes this.
- Largest model 2.8B (the paper: 8B); the RunPod tier would test a 6–8B model.

## All summary values

<!-- table:summary -->
| Key | Value |
|---|---|
| `model` | pythia-2.8b |
| `layer` | 21 |
| `top_periods` | 150, 149, 148, 147, 146, 145, 144, 143 |
| `n_base10_in_top8` | 0 |
| `top_peak_periods` | 2, 10, 38, 58, 5, 7, 15, 17 |
| `n_base10_in_top8_peaks` | 3 |
| `r2_T10` | 0.202713 |
| `r2_T10_neighbours` | -0.0057 |
| `r2_T5` | 0.030588 |
| `r2_T2` | 0.395121 |
| `conf_layer` | 21 |
| `conf_valid_top` | 2, 10, 71, 39, 150, 43, 44, 64 |
| `conf_hits` | 2, 10 |
| `conf_n_hits` | 2 |
| `conf_p_perm` | 0.004975 |
| `conf_p_set` | 0.031797 |
| `conf_r2_T5` | -0.006958 |
| `conf_r2_T10` | 0.085932 |
| `conf_other_models` | pythia-410m L16: hits none; pythia-1.4b L16: hits [2] |
<!-- /table:summary -->
