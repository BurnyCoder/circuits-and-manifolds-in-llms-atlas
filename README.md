# Atlas of Circuits & Manifolds

**An interactive, cited field guide to the mechanisms found inside neural networks** — attention-head circuits,
curved representation manifolds, interpretable neurons, sparse-autoencoder (SAE) features, steering vectors,
attribution graphs and lenses — with the data **recomputed on open models** wherever possible.

🌐 **Website:** https://burnycoder.github.io/circuits-and-manifolds-in-llms-atlas/ · Author: Libor "Burny" Burian

> Status (2026-11-27): under construction. The framework and the first entries (induction heads, IOI, grokking) are
> being built; see [reports/plan.md](reports/plan.md) for the plan and [reports/planned-runs.md](reports/planned-runs.md)
> for the replication criteria fixed before running.

## What it is

- **In plain words:** researchers have found that language models (and image models) contain small, readable
  "mechanisms": two attention heads that copy repeated text, numbers stored on circles and added like a clock, a map of
  emotions, a direction that makes a model refuse. This site collects the most interesting ones, explains each in plain
  language and in technical detail, and lets you poke at an interactive visualization of it.
- **For experts:** each entry re-tests the original paper's key claims on open models (GPT-2, Pythia, Qwen3,
  TransformerLens toy models, InceptionV1, models trained here), with controls, baselines and 95% bootstrap CIs, and
  reports a pass/partial/fail verdict against a criterion fixed in advance. Closed-model findings (Claude) are faithful
  redraws from published numbers, shown next to open-model replications by others.

## How it works

```
open model / tiny model ──► src/atlas/entries/<entry>.py ──► site/public/data/<entry>.json ──► React + D3 site
   (TransformerLens,          (metrics, controls, CIs,          results/<entry>/replication.json      (GitHub Pages)
    HF transformers)           published-vs-ours verdicts)
```

## Quick start

```bash
# Python pipeline (uv, Python 3.12, CUDA 12.8 torch)
cp .env.example .env            # set cache paths; API keys are optional
uv sync --extra graphs
uv run python -m atlas.run --entries induction_heads,ioi,grokking   # or --entries all, add --quick for a smoke test
uv run pytest

# Website
cd site && npm ci && npm run dev    # http://localhost:5173
npm test && npm run build
```

## Repository layout

| Path | What |
|---|---|
| `src/atlas/run.py` | single entry point; runs entry modules in order |
| `src/atlas/entries/` | one module per atlas entry (`compute(ctx)`) |
| `src/atlas/common/` | env/device, JSON export, logging, bootstrap CIs, attention scores, Fourier basis |
| `site/` | Vite + React + TypeScript website; entries are Markdown files in `site/src/content/entries/` |
| `results/` | raw metrics and replication verdicts per entry |
| `reports/` | plan, planned runs, literature review, experiments, reviews |
| `tests/`, `site/tests/` | pytest (synthetic ground truth) and Vitest (content: citations, quotes, data) |

## License

Code: MIT. Text is our own paraphrase with links to sources; quotes are short (< 15 words) and attributed.
