# Response to the methods review and the fact-check

Date: 2026-11-28. Reviews: [methods-review.md](methods-review.md) (verdicts, controls, statistics) and
[fact-check.md](fact-check.md) (citations, quotes, numbers attributed to papers). This file lists what was done for
each finding. Current numbers are in the regenerated tables ([README](../../README.md#results-so-far),
[experiments.md](../experiments/experiments.md)); none are typed here.

Status: **fixed** = code or text changed and the entry rerun where needed · **flagged** = row marked post-hoc (shown,
not counted; list in [`src/atlas/common/post_hoc.py`](../../src/atlas/common/post_hoc.py)) and the write-up corrected ·
**to do** = on the README to-do list.

## Methods review

| ID | Finding (short) | Status | What changed |
|---|---|---|---|
| C1 | attribution-graphs tuned across three full runs; criteria post-dated | flagged; to do | Both rows post-hoc; report rewritten (no curated supernodes in 2/3 graphs, no negative control). New pre-stated test with a wrong-intermediate control is on the to-do list. |
| C2 | sae-manifolds metric changed after a full run; flat curve | flagged; to do | Both rows post-hoc; report and site summary state the flat curve. Seeds per k and a bootstrap peak test: to do. |
| M1 | "random-init" controls were all-zero networks | fixed | `init_weights = True` on a copied config; induction-heads and othello-gpt rerun. Othello's control rises to about 72%; both verdicts unchanged. |
| M2 | manifold-steering layers changed after failing runs; via-intermediate nearly guaranteed | flagged; fixed (partly); to do | Both rows post-hoc; endpoint check now in code and reported for layers 12–26; site text and report corrected (6-D centroid span, layers 22–26). Piecewise-linear control: to do. |
| M3 | sae-features rule changed after smoke tests; "Golden Gate" is a bridge feature | flagged; fixed | Both rows post-hoc; Golden-Gate-specific rate reported separately (none of the steered completions names it); claim, summary and report renamed to a bridge feature. |
| M4 | calendar random-plane control 1–2 orders of magnitude too small | fixed | Random-plane displacement norm-matched; Wilson CI on the causal row; entry rerun. The causal row is also post-hoc (templates changed after the smoke test). |
| M5 | world-model isometry has no control and passes by construction | fixed | Untrained-encoder control added: it gives the same isometry, so the control row fails; page and report say the pass is uninformative. |
| M6 | calculator peak score counts T = 5 at R² ≈ 0; layer selected on the hypothesis | flagged; to do | Peak-score row post-hoc; report rewritten. R² floor, permutation null, fixed layer and pair-level split: to do. |
| M7 | number-helix patch not better than dimension-matched PCA-9 | fixed (write-up); to do | Report, site summary and README state that PCA-9 recovers as much. Random 9-D control and k-fold CV: to do. |
| M8 | CIs promised but computed in 3/26 entries | fixed (partly); to do | README and protocol no longer promise CIs everywhere; Wilson CIs added to refusal, calendar and ActAdd (`common.stats.wilson_ci`, tested). Remaining entries: to do. |
| M9 | verdict layer = argmax of test R²; margins within noise | fixed (write-up); to do | Linebreak and stories reports state the thin margins and that GPT-2 decodes the count as well as Qwen3. Validation-chosen layers and bootstrap CIs: to do. |
| M10 | site summaries assert claims our recomputation fails | fixed | Summaries of 12 entries now state our outcome; every entry page shows a generated "our recomputation" line; README results section rewritten. |
| M11 | J-lens rank row n = 3, arithmetic case removed after its ranks were seen | flagged; fixed | Row post-hoc; arithmetic ranks disclosed in the report and planned-runs; a confirmatory test on 24 fresh prompts was pre-stated (commit `8d1376e`) before it ran. |
| M12 | emotions dose-sweep pass has little weight | fixed (write-up); to do | Report explains the dependence of the two slopes; the "one-dimensional" sentence is now a hypothesis. Benign control and permutation test: to do. |
| M13 | BSF curve-ring claim cannot fail; toy evaluated out of distribution | fixed (partly); to do | Random-channel control added: it fails (random channels nearly as harmonic). Page and report updated. Random-init network and superposed toy evaluation: to do. |
| M14 | refusal random-direction ablation cannot fail; substring judge | fixed (partly); to do | Label-shuffled difference-of-means control added (passes); Wilson CIs. Text-free coherence check: to do. |
| M15 | grokking-runs rows 1–3 had no pre-stated criterion | flagged | Rows marked descriptive (post-hoc). |
| m1 | IOI faithfulness keeps heads at all positions | fixed (write-up); to do | Report states the leniency; per-position nodes: to do. |
| m2 | curve-detector control includes silent neurons | fixed (write-up) | Report notes it. |
| m3 | calendar random-word pool small; degenerate cyclic score | to do | — |
| m4 | induction random-head control is one draw | to do | — |
| m5 | copy-suppression behaviour has no control; compensation untested | fixed (write-up) | Compensation stated as a hypothesis; to do: control token. |
| m6 | colour-bicone claim 1 weakly discriminating | fixed (write-up); to do | Report says claim 2 is the real test; random-init VAE: to do. |
| m7 | helix ratio test can pass when full-layer recovery ≈ 0 | to do | Not triggered in current results. |
| m8 | protocol said trained models use 3 seeds | fixed | Protocol corrected. |
| m9 | belief-fractal deviations undocumented | to do | — |
| m10 | ActAdd published number source | resolved | The fact-check confirmed the > 90% vs 2% wedding figure in the source. |
| m11 | amendments committed after runs started | fixed (process) | Disclosed in planned-runs; `atlas.run` now records `src_dirty_at_start` in every result's metadata. |
| m12 | greater-than circuit-only keeps all MLPs clean | fixed (write-up) | Report says faithfulness is inflated relative to path patching. |

## Fact-check

All 22 rows were applied to the entry files (commit `3e1964d`), including the overstated "replicated by others" rows
(now partial or inconclusive), the J-lens 59% number (now attributed to the category swap; France → China is 76/192
top-1), three quotes re-attributed to the Goodfire blog posts, and citation metadata. The same J-lens number was also
corrected in the published column of the jlens verdict rows.
