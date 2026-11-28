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
<!-- /table:replication -->

## Interpretation

Weak evidence for a mod-10 code, not for base-10 circles. The amended peak-score row is post-hoc (added after the
pre-stated ranking failed) and is excluded from the tally. Even on its own terms it is weak: one of its three base-10
"peaks", T = 5, has held-out R² near zero (see the per-period curves on the site), so the pass rests on T = 2 (parity) and T = 10 (R² about
0.2 in late layers). Non-base-10 periods (4, 7, 15) also recur among the peaks, the peak score has no null, the verdict
layer is the one that maximises R²(2) + R²(5) + R²(10), and only two of four layers give three hits. The pre-stated
global ranking fails because periods longer than the range of a+b act as a monotone magnitude code. Details: [methods review](../../reviews/methods-review.md), M6.

## Limitations

- The paper's 28-neuron ablations and DAS are not reproduced; probes are correlational.
- The train/test split is by row, so some test prompts repeat training prompts or their commuted form (b + a).
- To do: count a peak only if R²(T) > 0.05, add a permutation null for the hit count, fix the layer a priori and split
  by unordered pair {a, b}.

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
<!-- /table:summary -->
