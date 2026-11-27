# Progress

Updated 2026-11-27 19:40 (local time, WEST). Plan: [plan.md](plan.md) · pre-stated criteria: [planned-runs.md](planned-runs.md) ·
results: [experiments/experiments.md](experiments/experiments.md).

## Status at a glance

| Phase | State |
|---|---|
| Repo, environment, site framework, Pages deploy | ✅ done |
| Literature + state-of-the-art review | ✅ done (fact-check pass pending at the end) |
| Plan + run matrix with pre-stated criteria (batches 1–5) | ✅ committed before each batch ran |
| Batch 1–2 full runs (induction, IOI, copy suppression, greater-than, grokking ×3 seeds, companion runs, Othello-GPT, curve detectors, attribution graphs) | ✅ 9 entries, 31/31 claims pass |
| Closed-model redraws (Biology, linebreaks, Scaling Monosemanticity, emotions, global workspace, persona vectors, phylogeny) | ✅ pages + interactive simulations written |
| Model Explorer (clickable GPT-2, Qwen3, Othello-GPT, InceptionV1, closed models, General LLM / General transformer views) | ✅ done; thumbnails for all entries added |
| Batch 3–5 smoke tests (17 modules, `--quick`) | 🔄 running |
| Batch 3–5 full runs | ⏳ next (est. 3–6 h of GPU time) |
| Experiment reports + learnings | 🔄 batches 1–2 written; 3–5 after runs |
| README rewrite, docs | 🔄 docs written; README after batch 3–5 results |
| HF upload of trained models (grokking ×3, Mess3, world model) | ⏳ after batch 3–5 |
| RunPod paper-scale runs (≤ $20) | ⏳ needs `RUNPOD_API_KEY` from the user |
| Fact-check, link check, NeurIPS-style self-review | ⏳ last |

Roughly 55% of the plan is done. Remaining: batch 3–5 runs and reports, README, HF upload, RunPod tier, review.

## Positive results so far

All 31 pre-stated claims in batches 1–2 pass (see the scoreboard). The closest published-vs-ours matches are IOI's
baseline logit difference and name-mover attention.

## Negative / weaker-than-published results so far

- Othello-GPT: the linear board edit is causal but much weaker than Li et al.'s.
- Copy suppression: L10H7's direct effect is negative on every prompt, but ablating it raises the copied token on only
  5/8 prompts (compensation by other heads).
- These are reported as such on the pages. Both claims' pre-stated criteria still pass.
