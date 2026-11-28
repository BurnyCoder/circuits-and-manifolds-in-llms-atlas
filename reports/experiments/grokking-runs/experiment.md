# Experiment: cross-seed grokking statistics (companion runs)

Code [`src/atlas/entries/grokking_runs.py`](../../../src/atlas/entries/grokking_runs.py) · site data
`site/public/data/grokking-runs.json` (shown in the grokking entry) · fidelity 🔵 re-aggregated from the companion
project.

## Question

Do grokking and the Fourier circuit hold across many seeds, and do the controls (no weight decay, random labels) fail
as expected? Do Nanda et al.'s progress measures (restricted loss falls before test loss; excluded loss rises during
circuit formation) hold in every seed?

## Setup

Source: the companion project [visualizing-grokking-modular-addition-in-transformer](https://github.com/BurnyCoder/visualizing-grokking-modular-addition-in-transformer)
(MIT), whose 22 runs use Nanda et al.'s settings: 11 seeds with weight decay 1, 5 with weight decay 0, 3 random-label
controls, 3 with the paper's split and different inits. We download only small per-run files from the Hugging Face
model repository `BurnyCoder/grokking-modular-addition-runs` (`meta.json`, `curves.npz`, `analysis/{phases,final}.json`,
and for the 11 weight-decay seeds `analysis/ckpt_metrics.npz`, ~12 MB each), not the ~364 MB checkpoint files, and
compute: grokking delay (test-95% step ÷ train-99% step), key frequencies per seed, fraction of neurons with > 85%
variance explained by one frequency, and the controls' final test accuracy.

**Progress measures (added 2026-11-27).** From the 401 per-checkpoint metrics of each weight-decay seed we re-implement
the companion's preregistered H1/H2 definitions (its `docs/operationalizations.md`):
- **H1:** the restricted loss (logits projected onto the key frequencies, all pairs) reaches 10% of its value at t1
  before the test loss does.
- **H2:** the excluded loss (key frequencies removed, train pairs) at t2 is more than 10× its value at t1.

The phase boundaries t1 (end of memorization), t2 (start of generalization) and t3 (end of cleanup) come from the
companion's `phases.json`. The criteria (≥ 10 of 11 seeds) were written in `reports/planned-runs.md` before running.
This is **not a blind test**: we had read the companion's verdict table (11/11 for both) beforehand, so the check
verifies the analysis code and definitions, not the phenomenon independently. The companion's H3–H5 verdicts are read
from its `reports/tables/hypotheses.json` and shown on the site, not re-tested.

## Results vs published claims

<!-- table:replication -->
| Claim | Published | Ours | Criterion | Verdict |
|---|---|---|---|---|
| Grokking replicates across seeds (companion project's 11 runs) | grokking with wd = 1 on a 30% split (Nanda et al. 2023, one run) | 11/11 seeds reach > 99% test accuracy; test-95% step ÷ train-99% step: median 57.2× (min 32.2×) | ≥ 9 of 11 seeds grok, with delay ≥ 3× in each | ✅ pass (post-hoc: descriptive row: no pre-stated criterion; companion results known in advance) |
| A handful of key frequencies and single-frequency neurons in every seed | 5 key frequencies; 84.6% of neurons single-frequency | key frequencies per seed: 3–6; median 81.6% of neurons > 85% FVE | 3–7 key frequencies in every seed and median single-frequency fraction > 60% | ✅ pass (post-hoc: descriptive row: no pre-stated criterion; companion results known in advance) |
| Controls: no weight decay / random labels don't generalize | no grokking without regularization; nothing to learn with random labels | wd = 0: best final test acc 21.5% over 5 runs; random labels: 0.9% over 3 | every wd = 0 run < 50% and every random-label run < 5% test accuracy | ✅ pass (post-hoc: descriptive row: no pre-stated criterion; companion results known in advance) |
| Restricted loss falls before test loss (progress measure; companion H1, cross-check) | restricted loss drops during circuit formation, before test loss (§5.1, Fig. 7); companion: 11/11 seeds | 11/11 seeds; restricted loss reaches 10% of its t1 value 700–2100 steps before test loss; 11/11 seeds match the companion's per-seed steps exactly | earlier in ≥ 10 of 11 seeds | ✅ pass |
| Excluded loss rises during circuit formation (progress measure; companion H2, cross-check) | excluded loss rises while train and test loss barely move (§5.1); companion: 11/11 seeds | 11/11 seeds; excluded loss at t2 is 12.6× to 7.89e+05× its value at t1 | > 10× in ≥ 10 of 11 seeds | ✅ pass |
<!-- /table:replication -->

## Interpretation

Rows 1–3 are descriptive: no criterion was written for them before the run and the companion's results were known, so
they are flagged post-hoc and not counted. They show that every weight-decay-1 seed groks with a delay of tens of × the
memorisation time, each with 3–6 key frequencies and mostly single-frequency neurons; no weight-decay-0 run generalises
and random labels stay at chance (1/113 ≈ 0.9%).

The progress measures (rows 4–5, pre-stated) behave as Nanda et al. describe in every seed: the Fourier circuit
(restricted loss) is in place hundreds to thousands of steps before the test loss collapses, and the training data's
reliance on the key frequencies (excluded loss) grows by at least an order of magnitude during the "flat"
circuit-formation phase. Our independent code gives exactly the companion's per-seed step counts. That is a code-level
cross-check of both implementations, not an independent replication. The one preregistered companion test that fails
is the strict key-frequency ablation (H4). That reflects how key frequencies are named (a weight-norm threshold), not
a missing circuit.

## Limitations

- These are the companion project's measurements re-aggregated, not independent training (our own 3 seeds are in
  `grokking`).

## All summary values

<!-- table:summary -->
| Key | Value |
|---|---|
| `n_runs` | 22 |
| `n_h1` | 11 |
| `n_h2` | 11 |
| `n_prog` | 11 |
| `h1_lead_min` | 700 |
| `h1_lead_max` | 2100 |
| `h2_ratio_min` | 12.610611 |
| `h2_ratio_max` | 789449.338153 |
| `agree_companion` | 11 |
| `comp_h3` | 11/11 |
| `comp_h4` | 6/11 |
| `comp_h4_supported` | no |
| `n_wd1` | 11 |
| `n_wd1_grokked` | 11 |
| `delay_median` | 57.214286 |
| `delay_min` | 32.222222 |
| `nkey_min` | 3 |
| `nkey_max` | 6 |
| `frac_neurons_median` | 0.816406 |
| `n_wd0` | 5 |
| `wd0_max_test` | 0.214677 |
| `n_rand` | 3 |
| `rand_max_test` | 0.008838 |
<!-- /table:summary -->
