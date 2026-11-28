# Progress

Updated 2026-11-28 01:20 (local time, WET). Plan: [plan.md](plan.md) · pre-stated criteria and dated amendments:
[planned-runs.md](planned-runs.md) · results: [experiments/experiments.md](experiments/experiments.md) · live site:
https://burnycoder.github.io/circuits-and-manifolds-in-llms-atlas/

## Status at a glance

| Phase | State |
|---|---|
| Repo, environment, site, GitHub Pages | ✅ live |
| Literature + state-of-the-art review | ✅ all external citation URLs resolve; content fact-check of the 33 entries done, 22 fixes applied ([reviews/fact-check.md](reviews/fact-check.md)) |
| Pre-stated criteria + dated amendments (batches 1–5) | ⚠️ the methods review found several analyses changed after results had been seen; those rows are flagged post-hoc and excluded from the tally ([planned-runs.md](planned-runs.md#post-review-disclosures-and-control-fixes-2026-11-28)) |
| Full runs: all 26 recomputed / re-aggregated entries | ✅ results in `results/`; scoreboard in README and experiments.md |
| Experiment reports + learnings for every entry | ✅ |
| Closed-model redraws (7) | ✅ |
| Model Explorer (General neural network / LLM / transformer views + 20 models) | ✅ |
| Hugging Face collection (grokking ×3, Mess3, world model) | ✅ |
| RunPod paper-scale tier (≤ $20) | ⏳ launcher and spend guard ready; needs `RUNPOD_API_KEY` in `.env` |
| Fact-check, adversarial methods review in `reports/reviews/` | ✅ done; controls fixed (random-init, norm-matched, label-shuffled, untrained encoder, random channels) and 8 entries rerun |
| Review follow-ups (CIs everywhere, larger n, new pre-stated tests for post-hoc rows) | ⏳ Wilson CIs started (refusal, calendar); rest on the README to-do list |
| Stretch entries | ⏳ |

Roughly 85% of the plan is done. Remaining: the review follow-ups (CIs, larger samples, new pre-stated tests), the
optional RunPod tier, and the stretch entries.

**Negative results from the review reruns:** two new controls fail and weaken earlier passes. Random InceptionV1
channels are nearly as orientation-harmonic as the curve neurons (block-sparse featurizers), and an untrained encoder
gives the same spline isometry as the trained mountain-car world model. Othello-GPT's corrected random-init control
rises to about 72% but the trained model's 98.9% still passes; refusal's label-shuffled control passes.

## Results

Generated tables: [README scoreboard](../README.md#results-so-far). The README's "In plainer language" section lists
what held up and what did not, with a link to each entry report.
