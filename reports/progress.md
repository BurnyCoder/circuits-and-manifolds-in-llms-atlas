# Progress

Updated 2026-11-27 20:00 (local time, WEST). Plan: [plan.md](plan.md) · pre-stated criteria and dated amendments:
[planned-runs.md](planned-runs.md) · results: [experiments/experiments.md](experiments/experiments.md) · live site:
https://burnycoder.github.io/circuits-and-manifolds-in-llms-atlas/

## Status at a glance

| Phase | State |
|---|---|
| Repo, environment, site framework | ✅ done |
| GitHub Pages | ✅ live since 2026-11-27 19:46 (first deploy had failed on missing data files; fixed with `status: pending`) |
| Literature + state-of-the-art review | ✅ done (link check: all 389 external URLs resolve; content fact-check pending) |
| Pre-stated criteria (batches 1–5) + amendments | ✅ committed before each full run |
| Batch 1–2 full runs (9 entries) | ✅ 31/31 claims pass |
| Closed-model redraws (7 entries) | ✅ live; linebreak simulation corrected to reproduce ringing |
| Model Explorer + thumbnails | ✅ |
| Batch 3–5 smoke tests (17 modules) | 🔄 10 done, 7 to go (ActAdd waiting for the 6 GB GPT-2 XL download) |
| Bugs found by smoke tests | ✅ fixed: ridge α selection, attention-sink outlier, helix minimum n, belief loss pairing, FLUX gating, HF token path |
| Batch 3–5 full runs | ⏳ next, est. 3–6 h GPU |
| Experiment reports + learnings | ✅ batches 1–2 · ⏳ batches 3–5 after full runs |
| README, docs, screenshots | ✅ (results tables regenerate from `results/`) |
| Hugging Face | ✅ grokking ×3 seeds uploaded to the collection · ⏳ Mess3 + world model after full runs |
| RunPod paper-scale tier (≤ $20) | ⏳ launcher + spend guard written; needs `RUNPOD_API_KEY` in `.env` |
| Fact-check, NeurIPS-style self-review | ⏳ last |

Roughly 60% of the plan is done. The full runs take the most remaining time (hours of GPU), followed by the reports and
review.

## Positive results so far

- Batches 1–2: all 31 pre-stated claims pass.
- Smoke tests (small samples, not final): calendar circles in cyclic order in GPT-2; refusal direction (ablation
  94% → 0% refusals; addition 0% → 100%); helix fit far above a shuffled control in Pythia-410m.

## Negative / weaker results so far

- Othello-GPT edit and copy-suppression ablation are weaker than published (both claims still pass).
- Smoke tests (final verdicts after full runs):
  - Day-arithmetic circle patching in Qwen3-1.7B works only on a minority of prompts.
  - Pythia ≤ 410M cannot do 2-digit addition, so helix patching cannot be tested there.
  - The Mess3 random-init control already decodes beliefs well (R² ≈ 0.93).
  - Goodfire-rule neuron sparsification costs more in dense Qwen3-1.7B than the open replication's gentler rule.
