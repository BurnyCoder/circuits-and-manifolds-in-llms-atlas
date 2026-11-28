# Replication protocol

How every 🟢 *recomputed* entry turns a published claim into a verdict.

1. **Claim selection.** For each paper we pick its 2–5 central, testable claims (usually the ones in the abstract or in
   a headline figure), written as one sentence each.
2. **Pre-stated criterion.** For each claim we write down, before running, the published value, our metric, the
   control or baseline, and a pass criterion. All of this goes in [`reports/planned-runs.md`](../reports/planned-runs.md),
   meant to be committed before the run. The methods review found several exceptions (criteria or analyses changed
   after full runs, or committed minutes after a run started); see step 6 and the post-review section of
   `planned-runs.md`. Criteria are deliberately coarse ("> 2× the control",
   "≥ 8 of 10 neurons") because our models, prompts and sample sizes differ from the papers'. They test whether the
   *phenomenon* holds, not whether a decimal matches.
3. **Controls.** Every causal claim gets a matched random control: random heads of the same count, random directions
   of the same norm, shuffled labels, or a random-initialised model. Every probe claim gets a random-init or
   shuffled-label baseline.
4. **Uncertainty.** Counts are reported as k/n. Only induction-heads, IOI and greater-than compute 95% percentile
   bootstrap CIs (`atlas.common.stats.bootstrap_ci`, 2,000 resamples); the other entries report point estimates, and
   several verdicts rest on small n (3–12 prompts per cell). Adding CIs everywhere is on the to-do list.
5. **Smoke test, then full run.** `--quick` runs find bugs. Bugs found in quick mode are fixed and logged in
   `reports/experiments/learnings.md`. A criterion should never change after results for that claim have been seen.
   Fixes of demonstrable bugs that do not depend on the outcome (tokenization, attention sinks, leakage, memory) are
   dated amendments.
6. **Verdict.**
   - ✅ pass: the criterion holds.
   - 🟡 partial: the module's pre-stated sub-criterion holds only in part.
   - ❌ fail: the criterion does not hold.
   - n/a: the claim could not be tested.
   - **post-hoc** (a flag, not a verdict): the metric, layer, selection rule or criterion behind the row was defined or
     changed after results for that claim had been seen, or no criterion was ever pre-stated. Post-hoc rows keep their
     pass/partial/fail label and are shown everywhere with the reason, but they are excluded from the pre-stated tally.
     The list is [`src/atlas/common/post_hoc.py`](../src/atlas/common/post_hoc.py).

   Verdicts appear on each entry page, on the Replications page, in `results/<slug>/replication.json`, and in the
   README scoreboard.
7. **Fidelity labels.**
   - 🟢 *recomputed*: our code, open model, our numbers.
   - 🔵 *replicated*: open-model work by others, re-rendered (e.g. public Neuronpedia graphs, companion runs).
   - 🟠 *redrawn*: closed-model findings re-drawn or simulated from the paper's published numbers, labelled as such.

## Known threats to validity

- Our prompts/datasets are re-created in the papers' style, not the original files (unless noted), so absolute numbers
  differ.
- Smaller models than the papers' (8 GB GPU): a failure on a 1.7B model is not a refutation of a 70B result.
- Single seeds for pretrained-model analyses (no training randomness). Of the trained-here models only grokking uses
  several seeds (3 here + 11 companion); belief-fractal, world-model, sae-manifolds and BSF use one.
- Verdict layers are often the argmax of held-out R² over all layers, with the control read at that layer; margins of
  a few hundredths (e.g. linebreak claim 1: 0.747 vs 0.70; claim 3 fails by 0.004) are within noise.
- Several claims per entry and many entries: we do not correct for multiple comparisons. Verdicts are descriptive
  replication checks, not hypothesis tests with a family-wise error guarantee.
