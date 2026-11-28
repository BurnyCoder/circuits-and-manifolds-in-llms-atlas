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
<!-- /table:replication -->

## Interpretation

Exploratory only; both rows are post-hoc. Three full runs moved the two-hop result from 0/3 to 1/3 to 3/3 while the
analysis code was being edited, and the criteria were written after the passing run. In two of the three graphs
(Oakland, Shanghai) there are no curated supernodes: the "intermediate" is any of the top-30 features whose
Neuronpedia auto-interp label contains "California" or "China" (e.g. a feature about news on economic or political
issues in China). The test accepts any positive path with up to one extra hop on each side and has no negative control,
so it is close to unfalsifiable as written. The multilingual comparison uses one graph per language and no null. These
are structural checks on graphs built by others, not interventions. Details: [methods review](../../reviews/methods-review.md), C1.

## Limitations

- No interventions (feature inhibition/injection) performed here.
- To do: pre-state a new test with a negative control (the same path search for a wrong intermediate, e.g. "Florida"
  in the Dallas graph, and for shuffled labels) and a fixed label set or curated supernodes.

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
<!-- /table:summary -->
