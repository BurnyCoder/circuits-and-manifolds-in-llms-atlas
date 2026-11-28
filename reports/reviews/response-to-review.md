# Response to the methods review and the fact-check

Date: 2026-11-28. Reviews: [methods-review.md](methods-review.md) (verdicts, controls, statistics) and
[fact-check.md](fact-check.md) (citations, quotes, numbers attributed to papers). This file lists what was done for
each finding. Current numbers are in the regenerated tables ([README](../../README.md#results-so-far),
[experiments.md](../experiments/experiments.md)); none are typed here.

Status: **fixed** = code or text changed and the entry rerun where needed · **flagged** = row marked post-hoc (shown,
not counted; list in [`src/atlas/common/post_hoc.py`](../../src/atlas/common/post_hoc.py)) and the write-up corrected ·
**re-tested** = a new confirmatory row was pre-stated in [`planned-runs.md`](../planned-runs.md) and pushed before it ran
(review follow-ups, commit `3090314`, 2026-11-28 04:40 UTC; code `b4059db`) · **to do** = on the README to-do list.

## Methods review

| ID | Finding (short) | Status | What changed |
|---|---|---|---|
| C1 | attribution-graphs tuned across three full runs; criteria post-dated | flagged; re-tested | Both original rows post-hoc. Confirmatory rows: the two-hop path is **not** specific to the intermediate under a label-permutation null (fail); the multilingual middle-layer overlap is far above a control triplet of unrelated graphs (pass). |
| C2 | sae-manifolds metric changed after a full run; flat curve | flagged; re-tested | Both original rows post-hoc. Six fresh seeds per k, k* chosen on three and tested on the other three with a two-level bootstrap: an interior optimum at k = 6 holds (pass); shattering never appears at small k (fail). |
| M1 | "random-init" controls were all-zero networks | fixed | `init_weights = True` on a copied config; induction-heads and othello-gpt rerun. Othello's control rises to about 72%; both verdicts unchanged. |
| M2 | manifold-steering layers changed after failing runs; via-intermediate nearly guaranteed | flagged; re-tested | Both original rows post-hoc, now on 112 paths per layer with CIs. Confirmatory row: a piecewise-linear path through the same centroids stays *closer* to the behaviour manifold than the spline at all three layers (fail), and visits the intermediate days as often. 64-D PCA with more prompts: to do. |
| M3 | sae-features rule changed after smoke tests; "Golden Gate" is a bridge feature | flagged; fixed | Both rows post-hoc; 48 completions per cell with Wilson CIs; the Golden-Gate-specific rate stays at zero. |
| M4 | calendar random-plane control 1–2 orders of magnitude too small | fixed | Random-plane displacement norm-matched; Wilson CI on the causal row. The causal row is post-hoc (templates changed after the smoke test). "Wrong rotation" control and a fixed layer: to do. |
| M5 | world-model isometry has no control and passes by construction | fixed | Untrained-encoder control: the trained − untrained contrast is negative with a rollout-bootstrap CI below zero, so the control row fails and the isometry pass is uninformative. |
| M6 | calculator peak score counts T = 5 at R² ≈ 0; layer selected on the hypothesis | flagged; re-tested | Peak-score row post-hoc. Confirmatory row with all fixes (fixed layer, unordered-pair split, R² > 0.05, label-permutation and period-set nulls): only T = 2 and T = 10 count (partial). The pair split halves R²(T = 10): the row split leaked duplicate and commuted prompts. |
| M7 | number-helix patch not better than dimension-matched PCA-9 | fixed; re-tested | Write-up states that PCA-9 recovers as much. Row 2 uses 5-fold CV with every basis max-selected (still partial). Confirmatory row: the helix beats a random 9-D subspace by a wide margin but not PCA-9 at either qualifying layer (partial). |
| M8 | CIs promised but computed in 3/26 entries | fixed (mostly) | Wilson or bootstrap CIs (over prompts, paths, documents, stories, games, rollouts, seeds × instances) now appear in the verdict rows of every entry except the following. attribution-graphs and calculator-probes report permutation p-values instead; block-sparse, grokking and grokking-runs report per-seed counts; belief-fractal and colour-bicone still report point estimates (to do). |
| M9 | verdict layer = argmax of test R²; margins within noise | fixed; re-tested | Confirmatory rows on fresh documents / stories with validation-chosen layers and bootstrap CIs: linebreak decodability and early-layer ratio both **inconclusive** (CI straddles the bar); stories decodability inconclusive, above a bag-of-words baseline; stories RSA fails. |
| M10 | site summaries assert claims our recomputation fails | fixed | Summaries state our outcome; every entry page shows a generated "our recomputation" line; summaries updated again after the follow-ups. |
| M11 | J-lens rank row n = 3, arithmetic case removed after its ranks were seen | flagged; re-tested | Row post-hoc; a confirmatory test on 24 fresh prompts (commit `8d1376e`) is partial (the J-lens wins most prompts, sign test just above p = 0.05). Swap rows now carry Wilson CIs. |
| M12 | emotions dose-sweep pass has little weight | fixed; re-tested | Confirmatory sweep (8 phrasings, sugar as benign control, permutation nulls): the afraid − calm slope is significant for Tylenol, but sugar rises almost as much, and the difference is not significant (partial). Preference shifts and the valence r carry bootstrap CIs. |
| M13 | BSF curve-ring claim cannot fail; toy evaluated out of distribution | fixed; re-tested | On superposed inputs over 3 seeds the BSF still beats TopK (pass). An untrained InceptionV1 is *more* orientation-harmonic than the trained one (control fails); the circular-shift null reproduces the observed share, as predicted. |
| M14 | refusal random-direction ablation cannot fail; substring judge | fixed; re-tested | Label-shuffled control (passes); Wilson CIs; text-free coherence check: ablated completions are fluent under the unmodified model (pass). Logs now withhold every non-refusing harmful completion. |
| M15 | grokking-runs rows 1–3 had no pre-stated criterion | flagged | Rows marked descriptive (post-hoc). |
| m1 | IOI faithfulness keeps heads at all positions | re-tested | With (head, position) nodes the circuit keeps slightly *more* of the logit difference (pass); the leniency hypothesis is not supported. |
| m2 | curve-detector control includes silent neurons | fixed | Responsive-control median and a bootstrap CI of the difference reported; verdict unchanged. |
| m3 | calendar random-word pool small; degenerate cyclic score | fixed | 93-word pool, NaN on tied angles (tested), and a geometric comparison (class-mean variance in 2 PCs) in favour of days and months. |
| m4 | induction random-head control is one draw | fixed | Control averages every same-size set of non-induction layer-1 heads; worst set reported; verdict unchanged. |
| m5 | copy-suppression behaviour has no control; compensation untested | re-tested | 24 prompts with a control token: the head always pushes the copied token down, but not more than the control token in a quarter to a half of prompts (partial); control tokens are often variants of prompt words. Compensation remains a hypothesis. |
| m6 | colour-bicone claim 1 weakly discriminating | re-tested | A random-init VAE has no bicone (control passes); a linear-RGB baseline aligns lightness and chroma but not hue. |
| m7 | helix ratio test can pass when full-layer recovery ≈ 0 | fixed | Row 3 now also requires full-layer recovery ≥ 0.5. |
| m8 | protocol said trained models use 3 seeds | fixed | Protocol corrected (and updated for the new seed sweeps). |
| m9 | belief-fractal deviations undocumented | fixed | Deviations table in the experiment report (d_head, optimiser). |
| m10 | ActAdd published number source | resolved | The fact-check confirmed the > 90% vs 2% wedding figure in the source. |
| m11 | amendments committed after runs started | fixed (process) | Disclosed in planned-runs; `atlas.run` records `src_dirty_at_start`; the follow-up runs started from a clean committed `src/`. |
| m12 | greater-than circuit-only keeps all MLPs clean | fixed (write-up) | Report says faithfulness is inflated relative to path patching. |

## Fact-check

All 22 rows were applied to the entry files (commit `3e1964d`), including the overstated "replicated by others" rows
(now partial or inconclusive), the J-lens 59% number (now attributed to the category swap; France → China is 76/192
top-1), three quotes re-attributed to the Goodfire blog posts, and citation metadata. The same J-lens number was also
corrected in the published column of the jlens verdict rows.
