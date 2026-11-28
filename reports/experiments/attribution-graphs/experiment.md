# Experiment: attribution graphs on Gemma-2-2B (public Neuronpedia graphs)

Code [`src/atlas/entries/attribution_graphs.py`](../../../src/atlas/entries/attribution_graphs.py) · site data
`site/public/data/attribution-graphs.json` · page `#/entry/attribution-graphs` · fidelity 🔵 re-rendered from open
replications.

## Question

Do the open Gemma-2-2B attribution graphs (circuit-tracer + GemmaScope transcoders, hosted on Neuronpedia) show the two
structural patterns from the Biology paper ([Lindsey et al. 2025](https://transformer-circuits.pub/2025/attribution-graphs/biology.html)):
two-hop factual recall through an intermediate concept, and language-independent features in the middle layers?

## Setup

- **Data source:** public featured graphs fetched read-only from Neuronpedia (`GET /api/graph/gemma-2-2b/{slug}`; we do
  not generate or upload graphs). Feature labels from `GET /api/feature/gemma-2-2b/{layer}-{source}/{index}`.
- **Pruning:** curated graphs keep their human-labelled supernodes; uncurated graphs keep the 30 features with the
  smallest influence (most important) score. An edge is kept if it is ≥ 5% of the strongest incoming edge of its target.
- **Two-hop test:** strength of the best positive path embedding → intermediate supernode → answer logit (product of
  edge weights) for the Dallas→Austin, Oakland, and Shanghai graphs.
- **Multilingual test:** Jaccard overlap of active features between the English, French and Chinese antonym graphs,
  split into early / middle / late thirds of the layers.

## Results vs published claims

<!-- table:replication -->
| Claim | Published | Ours | Criterion | Verdict |
|---|---|---|---|---|
| Two-hop factual recall goes through an intermediate concept | Dallas → Texas → Austin path plus a direct shortcut (Claude 3.5 Haiku) | positive embedding → intermediate → answer path in 3/3 public Gemma-2-2B graphs (Dallas, Oakland, Shanghai) | path present in all checked graphs | ✅ pass (post-hoc: criteria written after three runs had moved this result from 0/3 to 3/3) |
| Multilingual circuits share language-independent features in the middle | 20 of 27 features shared across English/French/Chinese; language-specific features early and late (Claude) | feature overlap (Jaccard of the 3 graphs): early 0.06, middle 0.17, late 0.07 | middle-layer overlap higher than both early and late | ✅ pass (post-hoc: criteria written after the runs had been seen) |
| Confirmatory: the two-hop path runs through the right intermediate | Dallas → Texas → Austin (Claude 3.5 Haiku); the path should not appear for a wrong intermediate | path specific to the intermediate (label-permutation p < 0.05) in 0/3 graphs (p: Dallas 0.122, Oakland 0.574, Shanghai 0.610); paths through wrong intermediates found in 0/9 checks | specific in all 3 graphs and wrong-intermediate paths in ≤ 1 of 9 (pre-stated 2026-11-28) | ❌ fail |
| Confirmatory: middle-layer overlap is specific to the multilingual graphs | language-independent features concentrate in the middle layers (Claude) | middle-band Jaccard: multilingual 0.17 vs unrelated-graph control 0.03 (multilingual early 0.06, late 0.07) | middle > early and late, and ≥ control middle + 0.05; partial if one holds (pre-stated 2026-11-28) | ✅ pass |
<!-- /table:replication -->

## Interpretation

Exploratory only; both rows are post-hoc. Three full runs moved the two-hop result from 0/3 to 1/3 to 3/3 while the
analysis code was being edited, and the criteria were written after the passing run. In two of the three graphs
(Oakland, Shanghai) there are no curated supernodes: the "intermediate" is any of the top-30 features whose
Neuronpedia auto-interp label contains "California" or "China" (e.g. a feature about news on economic or political
issues in China). The test accepts any positive path with up to one extra hop on each side and has no negative control,
so it is close to unfalsifiable as written. The multilingual comparison uses one graph per language and no null. These
are structural checks on graphs built by others, not interventions. Details: [methods review](../../reviews/methods-review.md), C1.

**Confirmatory tests (pre-stated 2026-11-28, commit `3090314`).** Two new rows test the same claims with controls.
- *Two-hop specificity fails.* Under a label-permutation null (1,000 random permutations of all feature-group labels),
  the correct-intermediate path is not stronger than the path through a randomly labelled group: see `label_null_p`
  in the summary values (none below 0.05). Wrong-intermediate labels produce no path (`wrong_paths_found` of
  `wrong_paths_total`), but mostly because those labels are absent from the graphs, so that half of the test cannot
  discriminate. Conclusion: in these graphs, positive multi-step paths from the city to the answer are common, and the
  path check cannot show that they run through the state or country concept. A real test needs interventions
  (inhibit the intermediate feature and check the answer), as in Lindsey et al.
- *Multilingual specificity passes.* The middle-band Jaccard of the three antonym graphs (`overlap_middle`) is well above
  that of three unrelated graphs (`overlap_ctrl_middle`) and above the multilingual early and late bands. With one
  graph per condition this is an existence check, not a population estimate.

## Limitations

- No interventions (feature inhibition/injection) performed here.
- The labels are Neuronpedia's auto-interp labels (fixed outside this analysis); only the Dallas graph has curated
  supernodes.
- One graph per language and condition; no uncertainty across prompts.

## All summary values

<!-- table:summary -->
| Key | Value |
|---|---|
| `n_graphs` | 9 |
| `two_hop_found` | 3 |
| `two_hop_total` | 3 |
| `overlap_early` | 0.057634 |
| `overlap_middle` | 0.170543 |
| `overlap_late` | 0.071895 |
| `error_share_median` | 0.244004 |
| `two_hop_specific` | 0 |
| `wrong_paths_found` | 0 |
| `wrong_paths_total` | 9 |
| `label_null_p` | Dallas 0.122, Oakland 0.574, Shanghai 0.610 |
| `overlap_ctrl_early` | 0.022031 |
| `overlap_ctrl_middle` | 0.026738 |
| `overlap_ctrl_late` | 0.003003 |
<!-- /table:summary -->
