# Methods review: adversarial NeurIPS-style audit of the replication verdicts

Date: 2026-11-28 · Scope: `src/atlas/entries/*.py`, `src/atlas/common/*.py`, `results/*/{metrics,replication}.json`,
`site/public/data/*.json`, `reports/planned-runs.md`, `reports/experiments/*`, `site/src/content/entries/*.md`,
`README.md`, `logs/*.log`, git history. Models were not re-run. Everything below was checked in code, saved data or logs
unless marked *(uncertain)*. Times: git uses CET (+0100), `meta.generated` and file names in `logs/` use UTC and local
CET as stated.

## Summary verdict

The atlas is unusually transparent for a project of this size. It has a dated amendment table, reports negative
results, flags quick runs, uses grouped/held-out splits in most probes, and keeps raw logs that made this audit
possible. Still, **the claim that every verdict uses a criterion fixed before the run does not hold**, and several
"pass" verdicts would not survive review:

- **Three entries were tuned after full runs.** In attribution-graphs, the analysis code was changed across three
  full runs, moving the result from 0/3 to 1/3 to 3/3, and the criteria were committed afterwards. In sae-manifolds,
  the claim-1 metric was replaced after a full run whose data already passed the new metric. In manifold-steering, the
  steering layers were changed after two full runs. The README states "No pass threshold was changed after a full run".
- **Two "random-init" controls are all-zero networks**, because TransformerLens pretrained configs set
  `init_weights=False` (induction-heads, othello-gpt).
- **Several controls are trivially easy** (calendar random plane, refusal random-direction ablation, BSF shuffled
  orientations), and one key claim has no control at all (world-model isometry).
- **Several write-ups misstate the data.** The "Golden Gate" feature is a generic bridge feature: 0/24 steered
  completions mention Golden Gate or San Francisco. The calculator's "base-10 peak" counts T = 5 at R² 0.03.
- **Only 3 of 26 entries compute the bootstrap CIs** that the README and protocol promise. Many verdicts rest on n = 3–12.

Recommended overall status: 57/75 "pass" is an over-count. At minimum, attribution-graphs (2 claims), sae-manifolds
claim 1, manifold-steering claim 2, sae-features claims 1–2 and calculator-probes row 2 should be relabelled
exploratory / post-hoc. The README summary needs rewriting accordingly.

## Findings

Severity: **critical** = verdict not valid as a pre-registered result; **major** = verdict, number or claim
misleading, or a control invalid; **minor** = weakens rigor, verdict likely unchanged.

| ID | Sev. | Entry · file:line | Problem | Why it matters | Concrete fix |
|---|---|---|---|---|---|
| C1 | critical | attribution-graphs · `attribution_graphs.py:89-93,109,160-176,230`; `logs/20261127_1900*.log`, `_190125`, `_190203` | Three full runs (`quick=False`) at 19:00, 19:01 and 19:02 CET changed the two-hop result from **0/3 to 1/3 to 3/3** while the code was edited (uncommitted). Changes: a per-target edge threshold, then top-30 features with Neuronpedia auto-interp labels. Criteria were committed in `aa0bb72` at 19:14, 11 min after the passing run (`meta.generated` 18:03 UTC, `git_commit d15ed50`). This is not in the Amendments table. In 2/3 graphs (Oakland, Shanghai) there are no supernodes, and the "intermediate" is any top-30 feature whose auto-interp label contains the substring "California"/"China" (e.g. "news articles referring to economic or political issues in China"). The test accepts any positive path with up to one extra hop on each side (`path > 0`) and has no negative control. | Textbook garden of forking paths: the verdict was reached by iterating the analysis until it passed. The test as written is close to unfalsifiable. | Relabel both claims "exploratory (criteria post-dated runs)". Pre-state a new test with a negative control: the same path search for a wrong intermediate (e.g. "Florida" in the Dallas graph) and for shuffled labels. Require curated supernodes or a fixed label set. Report all three runs in the amendments table. |
| C2 | critical | sae-manifolds · `sae_manifolds.py:227,245-252`; `planned-runs.md:129`; `logs/20261127_215834…log` l.90-100 | Claim 1's metric was changed from capture fraction to restricted-R² peak **after the first full run** (23:18–23:20 CET; amendment `3e2e8ed` 23:27). That run's log already shows restricted R² peaking at **k = 6 (0.643)**, so the amended criterion was known to pass before it was written. The final curve is flat: 0.42 / 0.49 / **0.53** / 0.49 / 0.52 / 0.52 / 0.48 / 0.48 for k = 3…25. That is 0.011 between k = 6 and k = 10, with one SAE seed per k and no CI. The amendment table header (`planned-runs.md:114`, "each made … before the corresponding full run") and README:161 contradict this. | The pass is post-hoc and within noise. "Degrades at both ends" (experiment.md) is not supported. | Mark claim 1 post-hoc. Report the original criterion's result (fail, after the bug fix). Run ≥ 3 SAE seeds per k with CIs, and pre-state a peak test (e.g. bootstrap over seeds and instances that R²(k*) − max(R²(3), R²(25)) > 0). Fix the amendment header and README line 161. |
| M1 | major | induction-heads `induction_heads.py:104`; othello-gpt `othello_gpt.py:138` | `HookedTransformer(model.cfg)` reuses a pretrained config with `init_weights=False` (TransformerLens `loading_from_pretrained.py:1777`), so parameters are uninitialised `torch.empty`. Rebuilding both configs here gave **all-zero** `W_E`, `W_pos`, `W_Q`, `W_O`. The saved data match: induction "random-init" scores are 0.027 for 13/16 heads, which is the uniform-attention value (≈ 0.0266). Othello "random-init" probe accuracy is exactly 0.539 at **every layer including the embeddings**, equal to the shuffled-label control. | Both "random-init" controls are constant networks, not random initialisations. "> 10× random-init" and "≥ 10 points above random-init" compare against chance by construction. README:124 ("random-init models give chance") reports the bug as a finding. The absolute thresholds probably still pass. | Build the control with `cfg = copy(model.cfg); cfg.init_weights = True`, or call `rand.init_weights()`. Assert the parameter std > 0. Re-run both controls and the README sentence. |
| M2 | major | manifold-steering · `manifold_steering.py:35,111,124`; `logs/20261127_211234…`, `_212649…`; `planned-runs.md:124` | (a) The layers changed from 12/16/20 to 22/24/26 **after two full runs**. The 21:12 run at L12/16/20 failed both claims (via-intermediate 0.00–0.05 vs 0.00). The 21:26 run at L22/24/26 started **before** amendment `a669be8` (21:35). The "endpoint check" that justifies the change is not in the code, logs or results. The original-layer results are reported nowhere. (b) The verdict layer is `min(E_man/E_lin)` (outcome-selected), and claim 2 is judged at that layer. (c) Claim 2 is nearly guaranteed by design: the periodic spline passes **exactly** through the intermediate centroids (at t = 0.5 for step 2, and t = 4/12 and 8/12 for step 3). Per the endpoint check, those centroids make the model say that day 92–100% of the time. A chord between vertices of a 6-D simplex never passes a third vertex. | The one "pass" mostly reflects how the steering paths are built, plus a layer choice made after seeing failures. It does not test the paper's claim that the data-driven manifold geometry matters. | Report the L12/16/20 full-run results. Commit the endpoint-check code and its output. Use a control that equalises interpolation, e.g. a piecewise-linear path through the same centroids (still visits them), and a 64-D PCA subspace as in the paper. Mark claim 2 post-hoc. |
| M3 | major | sae-features · `sae_features.py:38,44,119-125,185-213`; `logs/…193119…` (21:56), `…220729…`, `…222553…`; `planned-runs.md:128` | (a) The feature-selection rule changed twice after smoke tests whose **steering rates were logged**. Claim-2 counts were 2/5 (one focus token), then 1/5 with Golden Gate failing (three probes), then **3/5** (Neuronpedia search). The rule adopted (`f54feef`, 22:44) is the only one that passes, and its smoke run (22:25) predates the commit. (b) The selected "Golden Gate Bridge" feature #17608 is labelled "references to various bridges and bridge-related contexts". In the full run, **0/24** steered completions at α = 40/80 mention "golden gate" or "san francisco"; 20/24 contain "bridge" (Thames, guitar bridge, "bridgebuilders"). The keyword list counts plain "bridge" (l.38). The experiment.md text "steers most completions to the Golden Gate Bridge" is false. (c) Each cell has n = 12 completions and no CI. | The headline "Golden-Gate-style steering works in GPT-2" (README:141) is really generic bridge steering. The cross-concept pass rests on a selection rule picked after seeing steering outcomes. | Report all three selection rules' full-run results side by side. Rename claim 1 to "a bridge feature steers toward bridges". Score Golden Gate specificity separately (golden gate / san francisco only). Use ≥ 50 completions per cell with Wilson CIs. |
| M4 | major | calendar-circles · `calendar_circles.py:224-236`; `planned-runs.md:56` | The pre-stated control is "random 2-D plane, **same displacement**". The code instead moves by the class-mean difference projected onto a random plane, `R.T @ (m_rand[d+j] - m_rand[d])`. Its expected squared norm is 2/d ≈ 0.1% (d = 2048) of the full difference, whereas the circle-plane delta is the projection onto the top-2 PCs of the day activations. The control is therefore 1–2 orders of magnitude weaker, and 0% success is guaranteed. The partial verdict (30% vs 0%, n = 43; Wilson 95% CI ≈ [19%, 45%]) relies on it. | The criterion's control half is trivially satisfied. The causal claim is not tested against a matched perturbation. | Norm-match: rescale the random-plane delta to `‖P2.T @ (M[d+j] − M[d])‖`. Also report a "rotate by the wrong amount" control (j′ ≠ j) and CIs. |
| M5 | major | world-model · `world_model.py:13,170-193,233-240` | Claim 2 (isometry) fits the spline through **position-binned means indexed by the true position**. Arc length along any such curve is monotone in position by construction, so r(arc, Δpos) ≈ 1 almost regardless of what the encoder learned. The docstring's "Control: the same analysis on the untrained encoder" is run only for claim 1. That encoder already decodes position at R² 0.97, so it would very likely also pass claim 2. experiment.md: "The headline result replicates closely." | The pass has no discriminating power and no control, although the paper's statement is about learned geometry. | Run the isometry and void analyses on the untrained encoder and on a pixel-PCA baseline. Pre-state "trained r_arc − r_chord exceeds untrained by X". Report it as a fail if the untrained encoder matches. |
| M6 | major | calculator-probes · `calculator_probes.py:67-72,83,109,141-146` | (a) The amended peak-score row counts **T = 5 with R² = 0.03** (essentially zero; neighbours slightly negative) as a base-10 hit. The pass therefore rests on T = 2 (parity) and T = 10 (R² 0.20). There is no null for the peak score, and non-base-10 periods 4, 7 and 15 recur among the "peaks" in most layers and models. (b) The verdict layer is the one maximising R²(2) + R²(5) + R²(10), i.e. selected on the hypothesis' own periods and on test R². Only layers 21 and 24 of 4 give 3 hits; L8 and L16 give 2, so "consistently across layers" (experiment.md) is false. (c) The split is by row: 28/294 test prompts are exact duplicates of training prompts and 31 more are the commuted b + a (seed 0, recomputed). | The "new small-model evidence" in README:136-139 is a post-hoc metric that turns near-zero R² into peaks. | Require each counted peak to have R²(T) > 0.05 and a permutation null (shuffle s across prompts) for the hit count. Fix the layer a priori (e.g. ⅔ depth). Split by unordered pair {a, b}. |
| M7 | major | number-helix · `number_helix.py:107,208,236-255`; experiment.md | (a) "Linear-only recovers almost nothing, which supports the trigonometric (not magnitude) account" compares a 9-D helix patch with a **1-D** linear patch. The dimension-matched PCA-9 baseline recovers **81% vs helix 77%** at the reported layer 1 (L8: 71% vs 73%). In the paper the helix beats PCA (7.21 vs 6.13; Kantamneni & Tegmark Table 1, which uses PCA with 2k + 1 = 9 dims, checked). (b) The helix-fit layer is the argmax of held-out R² over 33 layers (20 test numbers, one split), while the shuffled control is read at that same layer, not max-selected. | The helix-specific causal claim is not supported over a generic 9-D subspace. The README calls this "new small-model evidence". | Report PCA-9 next to the helix in the write-up and README. Add a random 9-D subspace control. Use k-fold CV over a and report the max over layers for every basis, or fix the layer a priori. |
| M8 | major | cross-cutting · `common/stats.py` used only in `induction_heads.py`, `ioi.py`, `greater_than.py`; README:19; `docs/replication-protocol.md:15` | Bootstrap CIs exist in **3/26 entries**. Averages without any uncertainty include refusal rates, ActAdd and SAE steering rates, manifold E_BC, patch-recovery medians, probe R²s, and the Othello error. Strongly worded verdicts rest on tiny n: J-lens intermediates **n = 3** (one is a tie, eiffel 2 vs 2); J-lens swaps 1/8 vs 0/8 ("always more than random" in experiment.md); SAE steering n = 12 per cell; ActAdd n = 24; copy-suppression 8 hand-written prompts ("always pushes down"); manifold steering 28 paths from 7 source prompts. | Verdicts and wording cannot be checked against noise. The README's "95% bootstrap CIs" statement is inaccurate. | Add Wilson CIs for rates and bootstrap CIs for R²/E_BC (over stories, documents or prompts), and print them in `ours`. Increase n where cheap (steering, J-lens prompts). Soften wording when CIs cross the threshold. |
| M9 | major | linebreak-manifold `linebreak_manifold.py:118,142,145`; stories-in-space `stories_in_space.py:92,143`; number-helix `:107` | The verdict layer is the argmax of **test** R² over 9–33 layers, with no CI, and the margins are within noise. Linebreak claim 1 passes at 0.747 vs 0.70 on an erratic layer profile (L6 0.747 → L9 0.336 → L12 0.287), which suggests probe instability. Claim 3 **fails by 0.004** (0.668 vs 0.9 × 0.747 = 0.672), yet README:151 says the count "is not an early-layer phenomenon". Stories claim 1 passes at 0.543 vs 0.5 (3/9 layers ≥ 0.5). | Pass and fail here are coin flips presented as findings. | Choose the layer on validation documents or stories, or report max-over-layers for controls too. Add bootstrap CIs over documents/stories. Call results within the CI "inconclusive". |
| M10 | major | site entries `site/src/content/entries/{emotions,jlens,neuron-basis,stories-in-space,linebreak-manifold,sae-manifolds,sae-features,manifold-steering}.md:10`; README:123-161 | Reader-facing summaries assert the published claims that this atlas's own recomputation **fails**. Examples: "adding one changes the model's choices" (emotions, fail); "swapping its coordinates swaps the concept" (J-lens, both fail); "zero the rest and the model barely notices; … gate-open" (neuron-basis: +35.7% loss, quadrant claim fails); "rippled curve" (linebreak manifold claim fails); "manifolds shatter" (no shattering observed); "Golden Gate … Python feature … steers" (Python steering 0–8%). README:161 says "No pass threshold was changed after a full run" (contradicted by C1, C2, M2). README:129-135 lists the post-hoc or tautological passes (SAE sweet spot, isometric world-model manifold) as replicated geometry. `manifold-steering.md:14,39,56` still says layers 12–20 and 64-D PCA. | Readers will take recomputed entries as confirmations. Correctness of the summary text matters more than the tables. | Add a one-line "our replication: pass/partial/fail" next to each summary (generated from `replication.json`). Rewrite README:123-161 from the corrected verdicts. Fix stale site text. |
| M11 | major | jlens · `jlens.py:36-44,179-206`; `logs/…193119…` l.214; `planned-runs.md:120` | Dropping the arithmetic prompt from the rank metric (amendment 1) removed the only case where the lenses differ sharply, **in the logit lens's favour**. In the smoke run, the first-digit tokens of "21"/"42" ranked **1360/1088 (J-lens) vs 16/42 (logit lens)**. This is not mentioned in experiment.md. Including them would still give a pass (medians 6 vs 14, top-10 3/5 vs 2/5, recomputed). The remaining n = 3 includes a tie and a questionable intermediate ("Eiffel Tower" → France → Paris). experiment.md: "The lens claim replicates." | Omitted contrary evidence plus n = 3 cannot support "replicates". | Report the arithmetic ranks with the tokenization caveat. Add ≥ 20 two-hop prompts with single-token intermediates. Use "suggestive". |
| M12 | major | emotions · `emotions.py:137-145,223-229` | The dose-sweep criterion is the sign of two slopes over 7 single prompts, with no control prompt and no null. "afraid" and "calm" lie at opposite ends of the valence axis (PC1, \|r\| 0.94), so their slopes are strongly anti-correlated, and a joint pass by chance is closer to 50% than 25%. The number tokens themselves change with dose. experiment.md: "behaves as predicted". | The pass has little evidential weight. | Add a matched benign control ("{d} ml of water"), several paraphrases per dose, and a permutation test over dose order. Report the slope difference with a CI. |
| M13 | major | block-sparse-featurizers · `block_sparse.py:91-107,112-148` | (a) Curve-block claim: the stimuli vary only orientation (72) and radius (4), so the top-16 PCs of any reasonably smooth vision network are a periodic function of orientation, and low harmonics must dominate. The shuffled-orientation null (destroys smoothness) is trivially easy, and there is no random-init network control. (b) The toy per-concept R² shows each concept **alone**. That is the out-of-distribution evaluation the sae-manifolds amendment rejected ("TopK always fires k atoms"); BSF and TopK are evaluated inconsistently across entries. | The InceptionV1 pass cannot fail. The toy comparison method is inconsistent with the sibling entry's rationale. | Add a random-init InceptionV1 control and a smoothness-preserving null (e.g. circularly shifted orientation per channel). Evaluate the toy on superposed inputs as in sae-manifolds. |
| M14 | major | refusal-direction · `refusal_direction.py:40-41,143-151` | (a) The ablation control projects out a random unit direction, which removes about 1/d (d = 1536) of activation variance; "same norm" is irrelevant for ablation, so this control cannot fail. (b) "Refusal removed" is judged only by the absence of 12 refusal substrings in 40 tokens. Ablated outputs are withheld (reasonable for safety), so incoherent or degenerate text counts as compliance. experiment.md: "replicate strongly". | The ablation claim's control and measurement are both weak. | Control: ablate a matched-variance direction, e.g. the mean difference between two random halves of harmless prompts, or a direction with the same mean projection magnitude. Add a coherence check that stores no text (perplexity of the completion under the unablated model, or a safety classifier's label only). |
| M15 | major | grokking-runs · `grokking_runs.py:179-200`; `planned-runs.md:101-112` | Rows 1–3 (grokking across seeds, key frequencies, wd0/random-label controls) have **no pre-stated criteria**. planned-runs lists only the two progress-measure rows, and the author already knew the companion data. They count as 3 passes in the scoreboard. | Inflates the pre-registered pass count. | Label the rows "descriptive (not pre-stated)" and exclude them from the scoreboard, or move them to a separate tally. |
| m1 | minor | ioi · `ioi.py:201-202` | Faithfulness keeps circuit heads at **all** positions. Wang et al. define nodes as (head, token position) pairs (§2.1, p. 4, checked), e.g. name movers at END and induction heads at S2. The test is therefore more lenient than the paper's (93% vs 87%). | Faithfulness is overstated relative to the paper's definition. | Mean-ablate circuit heads outside their class positions (S1+1, S2, END). |
| m2 | minor | curve-detectors · `curve_detectors.py:71-77,135` | Silent control neurons get selectivity 0 by definition, which lowers the control median (0.10). | Easier control. | Restrict controls to neurons with non-trivial responses, or report both. |
| m3 | minor | calendar-circles · `calendar_circles.py:33,125-133`; `common/geometry.py:48-61` | The 40 random-word "control" sets are drawn from a pool of only 20 words and overlap heavily. A permutation p-value is calibrated under exchangeability by construction, so this control tests calibration, not specificity. `cyclic_order_score` returns 1.0 for all-equal angles (stable argsort), a degenerate pass. | Weak control; latent degenerate pass. | Use a larger word pool and a geometric control (do random words form a circle?). Return NaN on ties. |
| m4 | minor | induction-heads · `induction_heads.py:97-101` | The random-L1-head control is a single draw of 1 head (only one head, 1.6, exceeds 0.3). | Fragile control. | Average over all 7 non-induction heads or many draws. |
| m5 | minor | copy-suppression · `copy_suppression.py:169-175`; experiment.md | The behavioural claim has no control, e.g. L10H7's direct effect on non-copied tokens. Ablation (5/8) is not part of the criterion. "Negative name movers compensate" is untested speculation. | Overclaim from n = 8. | Add a control token per prompt; label the compensation remark a hypothesis. |
| m6 | minor | colour-bicone · `colour_bicone.py:81-88` | "3-D subspace" is nearly guaranteed for a 3-parameter input family. The noise-image control is uninformative. | Claim 1 is weakly discriminating (claim 2 is the real test). | Compare with a random-init VAE or a linear-RGB baseline. |
| m7 | minor | number-helix · `number_helix.py:254` | `helix ≥ 0.5 × full` passes trivially when full-layer recovery ≈ 0 (layer 16: 0.032 ≥ 0.5 × 0.063). Not triggered here. | Latent degenerate pass. | Require full-layer recovery ≥ 0.5 before the ratio test. |
| m8 | minor | `docs/replication-protocol.md:39` | "Trained-here models use 3 seeds": only grokking does. belief-fractal, world-model, sae-manifolds and BSF use one. | Inaccurate doc. | Fix the doc; add seeds where cheap. |
| m9 | minor | belief-fractal · `belief_fractal.py:107,113-124` | The Mess3 matrices match the paper's App. A.3 entries (0.765, 0.0675, 0.0425, 0.00375; checked). Undocumented deviations: d_head 64 vs the paper's 8, and Adam 30k × 512 vs SGD lr 0.01. | Minor fidelity gap. | Document it in Deviations. |
| m10 | minor | actadd · `actadd.py:3-8,29-30` *(partly uncertain)* | arXiv v5's running example is p+ = "weddings", p− = " ", l = 16, c = 1. The "I talk about weddings constantly" pair appears in its Table 1, but I could not find the "> 90% vs ~2%" figure in v5's main text (maybe from the AlignmentForum post). | Possible citation mismatch. | Cite the exact source of the published number. |
| m11 | minor | git order | Several amendments were committed minutes **after** the runs they enabled had started, e.g. day-arithmetic templates `c57cb29` 21:15 vs the manifold full run at 21:12, and Neuronpedia selection `f54feef` 22:44 vs its run at 22:25. "Git history shows the order" holds only loosely. | Weakens the audit trail. | Commit before every run; have the runner refuse to start with a dirty tree for `quick=False`. |
| m12 | minor | greater-than · `greater_than.py:98-101,122` | "Circuit-only" keeps all MLPs 0–11 clean (pre-stated; documented reason). This is more lenient than Hanna et al.'s path patching. | Faithfulness is inflated relative to the paper. | Note it next to the 92% figure. |

## Per-entry notes (entries with issues)

**attribution-graphs (C1).** The three `logs/20261127_1900*/_190125/_190203` runs show two-hop path strengths of
(0, 0, 0), then (6.58, 0, 0), then (6.58, 3.19, 3.26). The group counts jump from 14/13 to 44/43 for Oakland and
Shanghai when the "top-30 uncurated features" branch and auto-interp labels were added. The Oakland and Shanghai
intermediates in `site/public/data/attribution-graphs.json` are auto-interp features (e.g. L14 "mentions of China,
Korea, technology, and conflict"). They are not curated supernodes, contrary to the experiment.md limitation
("graph curation … is human-made"). The multilingual Jaccard (0.06/0.17/0.07) comes from one graph per language,
with no null.

**sae-manifolds (C2).** The bug disclosure in learnings.md is commendable. The fix, however, came with a criterion
change chosen with knowledge of the full-run curve. The final curve's maximum is within 0.011 of k = 10 and only
0.05 above k = 25. A sweet spot needs a decline on both sides beyond seed noise.

**manifold-steering (M2).** Besides the post-hoc layer change: `bc_dist` energies of 1.6–1.9 at every layer mean both
paths are far off the behaviour manifold. The pass on claim 2 coexists with a fail on the paper's main quantitative
claim. The docstring (l.15, "64-d component") and the site content are stale.

**sae-features (M3).** Learnings.md documents the label-based motivation for the rule change. It does not mention
that the smoke runs had logged steering rates for each rule, or that only the final rule passes claim 2.

**calendar-circles (M4, m3).** The GPT-2 geometry and SAE claims are fine as descriptive results. The SAE claim has no
control (latents selected on the same tokens).

**calculator-probes (M6).** Per layer (pythia-2.8b): peaks are [2, 10, 51, 4, 6, 7, 15, 13] at L8,
[2, 10, 50, 36, 7, 4, 15, 144] at L16, [2, 10, 38, 58, 5, 7, 15, 17] at L21 and [2, 10, 37, 5, 7, 15, 17, 29] at L24.
R²(5) is −0.03 / −0.02 / 0.03 / 0.01. T = 2 and T = 10 are the only consistent signals; T = 10 reaches R² ≈ 0.2 in
late layers. That is modest evidence for a mod-10 code, not for "base-10 circles".

**number-helix (M7, m7).** The n/a rule for < 20 solved prompts is good. Pythia-1.4B/2.8B solve 76–78% of prompts.
At layer 16 the full-layer recovery is ≈ 0.06, so information has left the a token by mid-depth. That is worth
stating, because "recovers most of a layer's effect" holds only for layers ≤ 8.

**othello-gpt (M1).** With a correct random init the control will likely rise well above 53.9%: occupancy is partly
decodable from token and position embeddings. Re-check the "≥ 10 points" margin (the black/white margin of 23 points
should survive). Headline accuracy includes empty squares; the occupied-only accuracy (97.9% best) is computed but not
reported in the verdict.

**induction-heads (M1, m4).** The claims themselves are well supported (ablation 2.68 → 12.45 nats, with CIs).
Only the control labels are wrong.

**linebreak-manifold, stories-in-space (M9).** The GPT-2 small "weak-model control" decodes the count at least as
well as Qwen3 (0.759 vs 0.747). The interpretation should say that decodability is not specific to the stronger
model. Stories claim 1 has no baseline (e.g. a bag-of-words ridge on the prefix text, or a shuffled-story control).

**jlens (M11, M8).** The swap tests are honestly failed. The pass on the lens claim needs more prompts.

**emotions (M12).** "In a 1.7B model the emotion space is effectively one-dimensional" does not follow from PC2's
low correlation with arousal (PC2 still holds 9% of variance). It is a hypothesis.

**world-model (M5).** The probe fail and the post-hoc GRU velocity result are reported honestly. The isometry
"replicates closely" claim is the problem.

**refusal-direction (M14).** The val/test separation with Arditi et al.'s splits is good. The addition claim
(6% → 98%, random 6%) has a norm-matched control and is the stronger of the two.

**grokking-runs (M15).** The exact per-seed match with the companion's H1 steps (11/11) is a good code-level
cross-check. It is not an independent replication.

## Things done well

- A dated amendment table with reasons. The original and amended analyses are both reported for J-lens (two swap
  variants) and calculator-probes (two rows).
- Negative results stay in the tables and on the site (15 fails).
- Leakage-aware splits: documents (linebreak), stories (stories-in-space), rollouts (world model, after the smoke
  test), sequences (belief fractal). Ridge α is chosen on grouped held-out training rows (`common/geometry.py:71-94`),
  never on test rows.
- `quick` outputs are flagged in metadata. Full prompts and outputs are logged, which made this audit possible.
  Harmful completions are withheld by design.
- Sensible guards: number-helix verdict n/a below 20 solved prompts; task-accuracy checks before steering
  (calendar, manifold); position ids from the attention mask for batched Qwen prompts.
- Mess3 transition matrices (checked against Shai et al. App. A.3), the IOI head classes and the TL-based scores
  follow the sources. The helix basis and PCA-9 baseline follow Kantamneni & Tegmark (checked against Table 1).

## How this review was produced

The review read every module in `src/atlas/entries/` and the common helpers, every `results/*/replication.json`, and
the per-layer arrays in `site/public/data/*.json`. It also used `planned-runs.md` and its git history, the
experiment and learnings files, and all run logs (matched against `quick=` flags and commit times).

Checks run:
- Rebuilt TransformerLens `attn-only-2l` and `othello-gpt` models from their pretrained configs and confirmed
  `init_weights=False` and all-zero parameters.
- Recomputed the calculator-probe split duplicates.
- Counted Golden Gate and San Francisco mentions in the steered completions logged in
  `logs/20261128_000152_run_sae_features.log`.
- Checked against the source papers: Wang et al. 2022 (§2.1, §4: (head, position) nodes, 87% faithfulness);
  Kantamneni & Tegmark 2025 (Table 1: full 8.34, helix 7.21, circle 6.83, PCA-9 6.13); Shai et al. 2024
  (App. A.3/A.6); Turner et al. arXiv v5 (Table 1, §4.1).

Sources: [Wang et al. 2022](https://arxiv.org/abs/2211.00593) ·
[Kantamneni & Tegmark 2025](https://arxiv.org/html/2502.00873) · [Shai et al. 2024](https://arxiv.org/html/2405.15943) ·
[Turner et al. (ActAdd) v5](https://arxiv.org/abs/2308.10248).
