# Plan — Atlas of Circuits & Manifolds in Neural Networks (approved 2026-11-27)

## Context
The user wants a public, interactive, heavily-cited website ("atlas") of the most interesting mechanistic-
interpretability findings — circuits, manifolds / neural geometry, neurons, SAE features, steering vectors,
attribution graphs — each clickable/expandable into a colorful, explained, interactive visualization, with "an
interactive LLM and other models" showing where circuits live. Everything sourced, quoted where possible.

User decisions (2026-11-27):
- **Hybrid data**: recompute real data from open models on the RTX 5070 Laptop (8 GB VRAM); closed-model findings
  become faithful interactive redraws/simulations, clearly labeled.
- **RunPod** allowed for models that don't fit 8 GB, **hard cap $20 total**.
- **`NEURONPEDIA_API_KEY` is in `.env`** (gitignored; never logged/committed).
- **Process**: atlas + replication reports (lit/SoTA review, plan, run matrix, src/tests, README, docs, per-entry
  experiment reports with honest pass/fail vs. published claims, self-review, GitHub + Pages, HF upload of models we
  train). No preregistration / paper.pdf.
- **Publishing**: public repo `BurnyCoder/circuits-and-manifolds-from-mechanistic-interpretability-in-llms-and-other-models-atlas`, GitHub Pages via Actions.
- Must cover: all links in the request, open-model **replications of closed-model results by others**, more
  mechinterp results, Anthropic emotions + global-workspace papers, Goodfire neurons/"already clean", and the whole
  **Goodfire Neural Geometry series** (6 posts).

Environment: folder contains only `.env`; not a git repo; uv, node 24, gh (BurnyCoder), WSL. C: has 31 GB free →
`HF_HOME`, `TORCH_HOME`, `UV_CACHE_DIR` on `D:\AI\...`. System Python 3.15 → uv-managed Python 3.12 venv in the project;
torch cu128 wheels (RTX 50xx).

## Constraints from research (7 research passes; full text → reports/literature-review.md)
- **Libraries (2026-11-27)**: TransformerLens 4.x removed `HookedTransformer`; circuit-tracer 0.5.5 pins TL `<4`,
  nnsight `<0.9`; SAELens 6.56 needs TL `>=3,<5` → one env pinned to latest TransformerLens 3.x (verify at install).
- **Ungated models first**: GPT-2 family (MIT), Pythia (Apache-2.0, checkpoints), Qwen3-0.6B/1.7B/8B, Qwen2.5-1.5B-
  Instruct (Apache-2.0), TL `attn-only-2l`/`othello-gpt`, Lucent InceptionV1 (Apache-2.0), FLUX.1-schnell VAE
  (Apache-2.0; verify same VAE as 2603.12261), OLMoE-1B-7B, Mistral-7B. Gated (Gemma-2/3, Llama-3.x, DINOv3): check
  `hf` access at implementation; if missing, give the user exact accept-license steps or use the ungated substitute.
- **Neuronpedia API (key in .env)**: graph generation on Gemma-2-2B (≤64 tokens, 30/h) → Biology-case graphs without
  local GPU; steering API (240/h) → Golden-Gate-style clamping on GemmaScope features; feature/activation endpoints.
  Uploading graphs to Neuronpedia = external publishing → ask first.
- **Licenses**: Distill CC-BY; CC-BY arXiv papers (figures reusable with attribution): 2603.12261 (LCS), 2308.10248
  (ActAdd), 2408.05147 (Gemma Scope), 2601.22594 (neuron circuits), 2605.01148 (calculator), 2604.28119 (SAE
  manifolds), 2605.12412 (stories in space), 2606.25234 (BSF), Scaling Monosemanticity arXiv copy 2605.29358 (verify).
  transformer-circuits.pub, arXiv-default papers: link + re-render from our data/schematics. **Goodfire website ToS
  forbids reproducing/scraping** → redraw from numbers, link out, never fetch atlas data in bulk. Code without a
  license (causalab, arithmetic-wild, TransluceAI/circuits) is read for reference only; we write our own code.
- **Quotes**: verbatim ≤15 words, **one per entry**, in that entry's own content file; all else paraphrased with
  section/figure deep links.

## Product design
**Home "Model Map" hero (the interactive LLM)**: clickable GPT-2 small (12×12 heads + MLPs on a residual spine)
colored by our measured per-head scores (induction, previous-token, duplicate-token, copy-suppression) with known
circuit roles overlaid (IOI classes, L10H7, greater-than heads/MLPs). Hover a head → real attention pattern + entries
using it. Tabs: 1L modular-addition model, Othello-GPT, InceptionV1, Qwen3-1.7B (linebreak/J-lens/neurons/emotions
layers), FLUX VAE, mountain-car world model, and closed models (Claude 3.5 Haiku / Sonnet 4.5) as black-box cards.

**Atlas grid**: filterable cards — category (Circuit / Manifold / Neuron / SAE feature / Steering / Attribution graph /
Lens), model, evidence (causal / correlational), fidelity badge (🟢 Recomputed here / 🔵 Open replication by others /
🟠 Redrawn/simulated), and **collection** (Anthropic Transformer Circuits thread, Goodfire Neural Geometry series,
Distill Circuits thread, Tegmark-lab geometry, …). Click → entry page.

**Entry page**: title · badge · one-line summary · layman + expert explanation (KaTeX) · interactive viz with controls,
legend, "how to read this" · key numbers (published = cited; ours = templated from JSON with 95% CI) · one quote ·
replications by others (✓ / partial / ✗) · our verdict vs pre-stated criterion · limitations · citations with deep
links · related entries. Other pages: Replications scoreboard, Methods glossary, Sources, About.

## Atlas entries
### A. Recomputed locally on open models (core)
| Entry | Model | Compute → viz | Controls / baselines |
|---|---|---|---|
| Induction heads | attn-only-2l, GPT-2 small, Pythia-160M checkpoints | repeated-random-token attention, prefix-match/copy scores, phase change across checkpoints | non-repeated tokens; random init |
| IOI | GPT-2 small | logit diff, per-head patching, name-mover attention, mean-ablation of all 2⁷ head-class combos → toggleable DAG | random-head knockouts |
| Copy suppression | GPT-2 small L10H7 | OV diagonal-rank histogram (MLP0(W_E)), QK test, worked prompts | random L10–11 heads |
| Greater-than | GPT-2 small | year×year prob heatmap, component-ablation toggles | random components |
| Grokking / modular add | 1L model trained here (P=113, d=128, 4 heads, 512 MLP, 30% split, AdamW 1e-3, wd 1; 3 seeds) | train/test curves, Fourier norms over training, key freqs, neuron heatmaps, embedding circles, scrubber | seeds; wd=0 |
| Othello-GPT | TL othello-gpt | mine/theirs linear probes per layer on self-generated games, board readout per move, probe intervention → legal moves | random-init probes; shuffled labels; black/white probe |
| Curve detectors (+ BSF block) | Lucent InceptionV1 mixed3b 379,406,385,343,342,388,324,340,330,349 | synthetic curves orientation×curvature → radial tuning; feature vis; block-sparse featurizer on curve responses → one orientation block, Fourier harmonics (paper: 89%) | straight lines; non-curve neurons; noise null |
| Calendar circles | GPT-2 small L7 (Engels MIT cluster indices + res-jb SAE), Qwen3-1.7B | day/month/year circles (3D) + how many SAE latents tile the circle (capture vs dilution) | random token sets |
| Number helix | Pythia 160M→2.8B (single tokens 0–99; verify) | helix T=[2,5,10,100] per layer, per-period circles, patching helix vs circle vs PCA vs polynomial | shuffled-order helix; 80/20 split |
| Linebreak counting manifold | Qwen3-1.7B (+ GPT-2 small control) | per-count mean residual PCA curve, probe R² by layer, cosine "ringing" heatmap | shuffled counts; T-Tech numbers |
| Belief-state fractal | 4L model trained here on Mess3 | residual→simplex fractal across training | random init; next-token regression |
| Toy superposition | live in browser (JS) | sparsity slider → phase change, polytopes, Dᵢ | — |
| Colour bicone | FLUX VAE (400 HSV solid colours) | top-3 PCs → HSL bicone in true colours | random-image PCA |
| SAE features | GPT-2 res-jb via SAELens (+ Neuronpedia embeds/API) | top activations on public text, logit effects, splitting across widths, steering by coefficient | random directions |
| ActAdd steering | GPT-2-XL | contrast-pair vectors, completions vs coefficient | zero / norm-matched random |
| Refusal direction | Qwen2.5-1.5B-Instruct | diff-of-means (repo datasets), per-layer PCA, refusal rate under ablation/addition | random direction |
| Emotion concepts | Qwen3-1.7B | ~10k stories (171 emotions × 20 topics × 3, emotion-word filtered), diff-of-means at ~⅔ depth minus neutral PCs → 171-point PC1/PC2 map, k=10 clusters, PC1/PC2 vs human valence/arousal norms, dose-sweep token heatmap, blissful/hostile steering on a 64-activity Elo test | shuffled labels; random-vector steering; no-neutral-PC ablation |
| Global workspace / J-lens | Qwen3-1.7B (pre-fitted `neuronpedia/jacobian-lens`), our own fit on Qwen3-0.6B | layer×position slice grid, token pinning, J-lens vs logit lens, occupancy, France→China swap, ablation selectivity | logit lens; random swaps/ablations |
| Neuron basis (Goodfire) | Qwen3-1.7B dense SwiGLU | per-token layer top-k over \|SiLU(g)·u\|, loss/KL-vs-k curve (k=480 ≙ 92.2%), band ablation, quadrant census + attribution share | random-k, bottom-k; vs ericzhou12 (KL≈0.80) |
| Attribution graph (open) | Qwen3-1.7B + circuit-tracer; Gemma-2-2B via Neuronpedia generate API | two-hop graph, pruning slider, feature inhibition; Biology-case graphs (Dallas, multilingual, rhyme, 36+59, known/unknown entity) in our D3 viewer | pruned completeness, error-node share |

Goodfire **Neural Geometry series** (one entry per post; blog = redraw, arXiv CC-BY figures reusable):
| Entry | Recompute | Compare to |
|---|---|---|
| The World Inside NNs (Geiger et al.) | mountain-car world model trained here (MountainCar-v0, 100 rollouts; conv enc 128²→64-d LN latent, 16-d action emb, GRU-64, residual conv decoder); position/velocity probes, spline through 100 position-bin means, manifold vs straight-chord interpolation with decoded frames, arc-length isometry | r=0.996 vs 0.06 (2605.05115 App. B); random-init encoder control |
| Steering Along Manifolds (Wurgaft et al. 2605.05115) | Qwen3-1.7B weekdays/months (check task accuracy first): periodic cubic spline through centroids in 64-d PCA, linear vs manifold steering, Hellinger behaviour path, isometry r, E_BC | weekdays r .99/.89, E_BC 0.34 vs 0.93; "Spokes, not a Ring" 2610.37796 counterpoint |
| A Geometric Calculator (Feucht et al. 2605.01148) | Fourier probes over periods 1–150 on "{a}+{b}=" across small models (Pythia; novel — no small-model evidence exists) + mod-2/5/10 clock simulation | 28 L18 neurons: zero → weekdays 91.8→40.8% |
| Can SAEs Capture Neural Geometry? (Bhalla et al. 2604.28119) | full synthetic benchmark: 8 manifold types × variants in d=128, TopK SAE 512 atoms, k sweep; restricted R², support size, receptive-field spread, Ising-coupling grouping → capture/shatter/dilution regimes; interactive manifold × k picker | capture peaks k≈4 |
| Meandering on Manifolds (Bigelow et al. 2605.12412) | SimpleStories (verify license) on Qwen2.5-1.5B-Instruct (Llama-3.1-8B-Instruct 4-bit if gated access): per-sentence 0–10 emotion ratings, layer-sweep ridge probes, valence–arousal trajectories | behaviour vs human valence–arousal r=.93 |
| Block-Sparse Featurizers (Fel et al. 2606.25234) | toy model (d=128, M=128, L₀=4) BSF vs TopK SAE per-block R²; InceptionV1 curve block (shared with curve entry); DINOv2 patch tokens (ungated stand-in for gated DINOv3) as stretch | BSF 0.93–0.97 vs TopK 0.53 |

Safety: never publish harmful completions (refusal entry shows rates + harmless-prompt refusals only); live persona/
emotion steering demos use sycophancy/optimism/blissful, not "evil".

### B. Redrawn/simulated (closed models), with open replications linked
| Entry | Redraw / simulation | Open replications |
|---|---|---|
| Biology of an LLM (8 case studies) | clickable supernode graphs with paraphrased labels + published intervention results | Neuronpedia Gemma-2-2B (Dallas ✓, multilingual ✓, rhyme partial, addition ✗); Jacopin 2604.15010/2609.18440 (plan at final prompt token); Ferrando 2411.14257 |
| Linebreaks (Claude 3.5 Haiku) | place-cell tuning, rippled-manifold particle sim, QK-rotation demo, decision hyperplane | T-Tech counting_manifolds (helix ~155-char period; early-layer peak ✗); statescope; our Qwen3 entry |
| Scaling monosemanticity / Golden Gate | splitting tree (SF 1→2→11), clamping explainer, safety-feature gallery + our Gemma steering via Neuronpedia | Eiffel Tower Llama; AxBench |
| Persona vectors | pipeline, projection-vs-trait scatter, preventative steering | LW extensions; assistant axis |
| Emotion concepts in Sonnet 4.5 | PC1 26%/PC2 15%, r .81/.66; blackmail/reward-hack steering slider; Elo shifts; transcript-viewer link | Hong, Hollowell, gemotions, 2606.26987, EmotionScope, Jeong, Jain et al. (calm ↑ hacking) |
| Global workspace in Claude | J_ℓ explainer, workspace band ~38–92, ~25-token occupancy, 21→42→49, swap 59% vs 5%, selectivity, J++ (55.2% vs 35.7%) | Viswanath (6.1% ✓), avalur (selectivity ✗ ≤9B), failed GPT-2 replication |
| Neurons as interpretable basis / already clean | sparsification table, quadrant causal share, Q3 replacement table, subcircuit→mode pipeline, eval scores, essay audit; critiques (Nanda, "is +2.9% sparse?") | Arora 2601.22594; Anthropic "Returning to the Neuron Basis"; ericzhou12; alexandertsai |
| Goodfire calculator (Llama-8B) / phylogeny (Evo 2) / world-inside (rhyme manifold) | simulations + published numbers | Marbut day-of-week CLT circle |

### C. Stretch (after A+B are live), priority order
Assistant axis (small Qwen3 recompute — none exists), J-space emergence across Pythia checkpoints, introspection
(honest mixed-evidence card), NLAs, attention sinks / massive activations, space & time probes, geometry of truth,
successor heads, function vectors, docstring circuit, high-low frequency detectors, multimodal neurons, entropy /
universal neurons, Park hierarchical geometry, SAE "crystals", weight-sparse transformers, emergent misalignment
direction, feature absorption, crosscoder diffing.

## RunPod tier (≤ $20 hard cap)
Scripts are debugged locally on small models first; pod time is pure compute. Jobs grouped by model to load each
once, on one 24–48 GB GPU (community/spot; check live price before launch; ~$0.4–0.8/h → ~25–50 h possible; plan
~15–20 h). Priority order (lowest dropped first if price is higher):
1. **Llama-3.1-8B** (if gated access, else Qwen3-8B/Mistral-7B): geometric calculator (Fourier probes, 28-neuron
   keep/zero/flip ablations), manifold steering (weekdays/months), Engels circular-probe interventions — 4–6 h.
2. **Pythia-6.9B / GPT-J-6B**: number helix at paper scale (helix 7.21 vs PCA 6.13 logit diff on GPT-J) — 3–4 h.
3. **Qwen3-8B**: emotion stories via vLLM (~40k) + extraction/steering; J-lens selectivity/swaps (pre-fitted lens);
   linebreak manifold at 8B (vs T-Tech) — 5–8 h.
4. **OLMoE-1B-7B**: MoE neuron sparsification (global top-k across routed experts) — 1–3 h.
5. Qwen2.5-7B-Instruct persona vectors; Gemma-2-2B circuit-tracer fallback only if the Neuronpedia API can't cover a case.
Spend controls: `src/atlas/runpod/` (runpod SDK) launches a pod, syncs code, runs the job list, pulls `results/`,
**terminates in `finally`**, appends rate × uptime to `reports/runpod-spend.md`, refuses to launch if logged + estimated
spend > $20. No network volumes. Needs from the user: RunPod account with credit + `RUNPOD_API_KEY` in `.env` (I never
enter payment details). Exact spend reported after each session.

## Technical architecture
```
circuits-and-manifolds-in-llms-atlas/
  pyproject.toml, uv.lock, .env(.example), .gitignore, README.md, AGENTS.md
  src/atlas/
    run.py            # wrapper: `uv run python -m atlas.run --entries all|ioi,induction --device cuda`
    common/           # io (compact JSON), models (cached loaders), stats (bootstrap CI), patching, probes,
                      # fourier, pca, splines, sae (TopK/BSF), neuronpedia client, logging (timestamped → logs/)
    entries/<slug>.py # compute(cfg) → results/<slug>/*.json + site/public/data/<slug>.json
    runpod/           # pod launcher + spend guard + remote job runner
    report.py         # regenerates README/report tables + scoreboard from results/ (no hand-typed numbers)
  tests/              # pytest: synthetic ground truth (helix fit, Fourier probe, spline steering, SAE regimes on a
                      # circle, BSF block recovery), Othello legality, patching on tiny model, JSON schemas
  site/               # Vite + React + TypeScript, hash router
    src/components/   # AtlasGrid, EntryPage, FidelityBadge, Quote, CitationList, ModelMap, Scoreboard
    src/viz/          # D3: AttentionArcs, HeadGrid, Heatmap, CircuitDAG, AttributionGraph, OthelloBoard, TuningCurve,
                      # LineChart, TrainingScrubber, SteeringSlider, SliceGrid (J-lens), ParticleManifoldSim,
                      # ToySuperposition, FramePairInterpolator; Plotly gl3d lazy: PointCloud3D, Curve3D
    src/content/entries/<slug>.md   # YAML frontmatter (title, category, collection, models, fidelity, evidence, viz,
                                    # data, citations[], quote, replications[]) + body; {{data.key}} templates
    public/data/<slug>.json         # budget < ~30 MB total
    tests/            # vitest: frontmatter schema, ≥1 citation URL, quote ≤15 words, data files exist, templates resolve
  results/<slug>/     # raw metrics + replication.json {claim, published, ours, ci95, criterion, verdict}
  reports/            # literature-review.md, state-of-the-art-review.md, plan.md, planned-runs.md, progress.md,
                      # summary.md, runpod-spend.md, experiments/{experiments.md, learnings.md, <slug>/…}, reviews/
  docs/               # architecture.md, data-pipeline.md, adding-an-entry.md, replication-protocol.md
  logs/               # timestamped; every generation prompt+output logged in full
  .github/workflows/pages.yml   # npm ci → vitest → build → deploy-pages
```
HF: models we train (modular addition ×3 seeds, Mess3 belief model, mountain-car world model) → one HF collection
"Circuits & Manifolds Atlas" under BurnyCoder (check `hf auth whoami`).

## Order of work (feature branches → PRs merged by me; no AI attribution in commits)
1. `git init`, `uv init` + deps, Vite scaffold, `.gitignore` (incl. `.env`), `.env.example`; create public repo; push.
2. `reports/literature-review.md` + `state-of-the-art-review.md` from the research (prior-art sites included);
   fact-check every item flagged unverified. Push.
3. `reports/plan.md`, `reports/planned-runs.md` (entry × model × data × metric × control × baseline × published value ×
   pass criterion), README v1. Push.
4. Site framework + 3 vertical slices (induction, IOI, modular addition), tests, Pages deploy. Merge.
5. Remaining section-A entries in batches (circuits → geometry → Neural Geometry series → features/steering/lens →
   attribution graphs); after each batch: experiment reports, README, data, push. HF uploads after training runs.
6. Section-B redraws/simulations, Model Map hero, Replications scoreboard.
7. RunPod session(s) (≤ $20) with batched jobs; terminate; log spend; add paper-scale numbers next to small-model ones.
8. Fact-check all claims + link-check every URL; NeurIPS-style self-review (stats, CIs, controls, no graph crimes,
   colorblind-safe palettes) → `reports/reviews/`; fix.
9. Section-C stretch entries as time allows; final README/docs/progress; final check against CLAUDE.md.

## Verification
- `uv run pytest` and `npm --prefix site run test` pass; `npm --prefix site run build` has no TS errors.
- `uv run python -m atlas.run --entries all` regenerates every data file; `uv run python -m atlas.report` regenerates all
  tables; logs inspected for warnings and full generation transcripts.
- Each replication judged against its pre-stated criterion in `planned-runs.md`; verdicts incl. failures on the
  scoreboard with 95% bootstrap CIs.
- Visual QA in the Claude browser pane (`preview_start` on vite preview): every entry, every control, console clean,
  light/dark, phone width; screenshots into README.
- Link check of all citation URLs; https://burnycoder.github.io/circuits-and-manifolds-from-mechanistic-interpretability-in-llms-and-other-models-atlas/ loads and deep
  links work; `reports/runpod-spend.md` total ≤ $20.
