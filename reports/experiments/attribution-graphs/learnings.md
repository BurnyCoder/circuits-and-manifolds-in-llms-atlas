# Learnings: attribution graphs

- **Do not publish on the user's behalf.** Neuronpedia's graph-generation API would create public graphs under the
  user's account, so we only fetch existing public graphs read-only.
- **Path detection needs per-target thresholds.** A global edge threshold found 0 two-hop paths because edge magnitudes
  vary by orders of magnitude across layers. Keeping edges ≥ 5% of each target's strongest incoming edge recovers them.
- **Labels.** Raw graph JSON has feature indices only. Neuronpedia's feature endpoint provides auto-interp explanations.
- **Uncurated graphs.** Graphs without supernodes are pruned to the 30 most influential features, which keeps the
  viewer readable.
- **Error nodes.** About a quarter of attribution flows through error nodes (`error_share_median`): the replacement
  model explains only part of the computation.
