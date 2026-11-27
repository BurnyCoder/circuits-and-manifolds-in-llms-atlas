# Replication protocol

How every 🟢 *recomputed* entry turns a published claim into a verdict.

1. **Claim selection.** For each paper we pick its 2–5 central, testable claims (usually the ones in the abstract or in
   a headline figure), written as one sentence each.
2. **Pre-stated criterion.** For each claim we write down, before running, the published value, our metric, the
   control or baseline, and a pass criterion. All of this goes in [`reports/planned-runs.md`](../reports/planned-runs.md),
   committed before the run (git history shows the order). Criteria are deliberately coarse ("> 2× the control",
   "≥ 8 of 10 neurons") because our models, prompts and sample sizes differ from the papers'. They test whether the
   *phenomenon* holds, not whether a decimal matches.
3. **Controls.** Every causal claim gets a matched random control: random heads of the same count, random directions
   of the same norm, shuffled labels, or a random-initialised model. Every probe claim gets a random-init or
   shuffled-label baseline.
4. **Uncertainty.** Means come with 95% percentile bootstrap CIs over prompts/examples (`atlas.common.stats.bootstrap_ci`,
   2,000 resamples) where the metric is an average; counts are reported as k/n.
5. **Smoke test, then full run.** `--quick` runs find bugs. Bugs found in quick mode are fixed and logged in
   `reports/experiments/learnings.md`. A criterion is never changed after seeing full-run results. If a method had
   to change (e.g. the greater-than circuit-only patching), the change and its reason are documented, and the
   criterion stays as written.
6. **Verdict.**
   - ✅ pass: the criterion holds.
   - 🟡 partial: the module's pre-stated sub-criterion holds only in part.
   - ❌ fail: the criterion does not hold.
   - n/a: the claim could not be tested.

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
- Single seeds for pretrained-model analyses (no training randomness); trained-here models use 3 seeds.
- Several claims per entry and many entries: we do not correct for multiple comparisons. Verdicts are descriptive
  replication checks, not hypothesis tests with a family-wise error guarantee.
