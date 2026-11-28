# Progress

Updated 2026-11-28 09:15 (local time, WET). Plan: [plan.md](plan.md) · pre-stated criteria and dated amendments:
[planned-runs.md](planned-runs.md) · results: [experiments/experiments.md](experiments/experiments.md) · live site:
https://burnycoder.github.io/circuits-and-manifolds-from-mechanistic-interpretability-in-llms-and-other-models-atlas/

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
| Review follow-ups (CIs everywhere, larger n, new pre-stated tests for post-hoc rows) | ✅ 2026-11-28: 18 confirmatory tests and 9 control fixes pre-registered (`3090314`), implemented (`b4059db`), smoke-tested and run in full (21 entries, clean `src/`); CIs in all entries except those listed in M8; status per item in [reviews/response-to-review.md](reviews/response-to-review.md) |
| Stretch entries | ⏳ |

Roughly 90% of the plan is done. Remaining: a few review items (CIs for belief-fractal and colour-bicone, attribution
interventions, a 64-D PCA steering subspace), the optional RunPod tier (needs `RUNPOD_API_KEY`), and the stretch entries.

**Confirmatory tests (2026-11-28, pre-registered before running):** counts are generated in the README scoreboard.
Passes: multilingual overlap above unrelated graphs, an SAE
sparsity optimum over held-out seeds, block-sparse featurizers on superposed inputs, fluent refusal-ablated text, IOI
faithfulness with (head, position) nodes, no colour bicone in a random-init VAE. Fails: the two-hop attribution path is
not specific to the intermediate; a piecewise-linear path through the same weekday centroids beats the manifold spline;
an untrained InceptionV1 is more orientation-harmonic than the trained one; no SAE shattering at small k; weak
behaviour–activation geometry on fresh stories. Partial or inconclusive: calculator (parity and mod-10 only), helix (beats
random 9-D but not PCA-9), emotions (the danger signal also rises for sugar), copy suppression (variants of the copied
word are suppressed too), linebreak ×2 and stories decodability (CIs straddle the bar).

**Negative results from the review reruns:** two new controls fail and weaken earlier passes. Random InceptionV1
channels are nearly as orientation-harmonic as the curve neurons (block-sparse featurizers), and an untrained encoder
gives the same spline isometry as the trained mountain-car world model. Othello-GPT's corrected random-init control
rises to about 72% but the trained model's 98.9% still passes; refusal's label-shuffled control passes.

## Results

Generated tables: [README scoreboard](../README.md#results-so-far). The README's "In plainer language" section lists
what held up and what did not, with a link to each entry report.
