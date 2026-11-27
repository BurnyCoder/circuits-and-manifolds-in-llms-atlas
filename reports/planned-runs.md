# Planned runs and pre-stated pass criteria

Date: 2026-11-27 · Author: Libor "Burny" Burian

Every recomputed entry tests specific published claims on open models. The pass criterion for each claim is written
here **before** the run that produces the number (git history shows the order). Criteria are also encoded in the entry
module (`src/atlas/entries/<name>.py`, `verdict_row(...)`), and results appear on the site's Replications page and in
`results/<slug>/replication.json`. Failures are reported, not tuned away; if a criterion is changed after seeing data,
the change and reason are logged in `reports/experiments/learnings.md`.

Conventions: 95% percentile-bootstrap CIs over prompts/sequences (2,000 resamples); seeds listed per run; GPU = RTX 5070
Laptop (8 GB) unless marked RunPod.

## Batch 1 — vertical slices (local GPU)

| Entry | Model / data | Claim | Published value (source) | Pass criterion | Controls / baselines |
|---|---|---|---|---|---|
| induction-heads | TL `attn-only-2l`; 128 seqs of 25 random tokens repeated | Induction heads form in layer 1 of a 2-layer attn-only model | need ≥2 layers (Elhage et al. 2021) | max layer-1 induction score > 0.3 and > 10× random-init score | random-init model, same config |
| induction-heads | same | Induction heads causally implement in-context copying | ablation removes ICL in small models (Olsson et al. 2022, Arg. 3) | zero-ablating induction heads raises repeat loss > 1 nat and > 3× the increase from ablating an equal number of random L1 heads | random L1 heads |
| induction-heads | same | Induction needs a previous-token head (K-composition) | Elhage et al. 2021 | ablating the top previous-token head raises repeat loss > 1 nat | — |
| induction-heads | Pythia-160M, 18 checkpoints (step 0 → 143k), 16 repeated-token seqs + 32×512 WikiText-103 tokens | Induction heads appear abruptly early in training | ICL jump at ~2.5–5B tokens in Anthropic's models (Olsson et al. 2022, Arg. 1) | largest between-checkpoint jump in max induction score happens before 20B tokens and is > 50% of the total rise | step-0 checkpoint |
| ioi | GPT-2 small; 8 own templates × 40 prompts (BABA/ABBA), p_ABC corruptions | GPT-2 small prefers IO over S | Δ = 3.56, IO > S in 99.3% (Wang et al. 2022) | mean Δ > 2 and IO > S in > 90% | p_ABC prompts |
| ioi | same | Name movers attend to IO | mean attention 0.59 | attention to IO > 2× attention to S | — |
| ioi | same | Negative name movers write against the answer | negative copy score 98% | DLA of 10.7 and 11.10 both negative | — |
| ioi | same | The 26-head circuit is faithful | 87% of Δ | circuit-only Δ within 60–120% of full (40–60% = partial) | mean-ablation from p_ABC per template |
| ioi | same | Circuit knockout breaks the task; random heads don't | §4 | all-class knockout < 30% of full Δ and random-26-head median > 60% | 40 random sets of 26 heads |
| grokking | 1L transformer trained here (p=113, d=128, 4×32 heads, 512 ReLU MLP, no LN, 30% train, AdamW lr 1e-3, wd 1, β=(0.9,0.98), 40k full-batch epochs); seeds 0,1,2 | Grokking: generalization long after memorization | train ~100% by ~1.4k epochs, test at ~10k (Nanda et al. 2023) | for every seed: epoch(test ≥ 95%) ≥ 3 × epoch(train ≥ 99%) and final test acc > 99% | wd = 0 run |
| grokking | same | Embedding is sparse in Fourier basis | ~6 non-negligible of 56 freqs | ≤ 10 freqs hold 90% of ‖W_E‖² in every seed | — |
| grokking | same | A few key freqs; most neurons single-frequency | 5 key freqs; 84.6% neurons > 85% FVE | 3–7 key freqs per seed and > 60% periodic neurons (seed 0) | — |
| grokking | same | Key frequencies causally necessary | Fig. 6 | removing key freqs from W_E → test acc < 20%; removing same number of random freqs → > 90% | 10 random draws |
| grokking | same, wd = 0, seed 0 | Weight decay needed for grokking in this setup | §5.3 | wd=0 final test acc < 50% | — |

## Batch 2 — circuits (local GPU; criteria written after the code-level smoke test, before full runs)

| Entry | Model / data | Claim | Published value (source) | Pass criterion | Controls / baselines |
|---|---|---|---|---|---|
| copy-suppression | GPT-2 small L10H7; 5,000 token ids (300–5,299), effective embedding W_E + layer-0 (self-attn approx.) + MLP0 | OV circuit suppresses the attended token | 84.70% diag in 10 most negative; 98.86% bottom 5% (column ranking; McDougall et al. 2023) | row-wise top-10 fraction > 50% and > 5× the median other L10–11 head | 23 other heads in layers 10–11 |
| copy-suppression | same | QK attends to predicted tokens | 95.72% diagonal-max | > 50% (20–50% = partial) | — |
| copy-suppression | 8 hand-written prompts | Lowers the copied token's logit on natural prompts | Table 1 examples | negative direct effect on ≥ 75% of prompts | zero-ablation of L10H7 |
| greater-than | GPT-2 small; 12 nouns × 97 start years 1702–1798 (single-token YY), corrupted start year 1701 | GPT-2 predicts years > YY | PD 81.7% (Hanna et al. 2023) | PD > 50% | corrupted prompts |
| greater-than | same | Heads 5.1/5.5/6.9/7.10/8.8/8.11/9.1 + MLPs 8–11 carry it | circuit keeps 89.5% | patching heads+MLPs drops PD < 30% of full **and** circuit-only (all 137 other heads patched, MLPs clean) keeps ≥ 60% (first condition only = partial) | — |
| greater-than | same | Random heads are not the circuit | (ours) | median PD with 7 random heads patched > circuit-heads-patched PD + 10 points | 30 random 7-head sets |
| othello-gpt | TL othello-gpt; 3,000 train / 500 test synthetic random full games (our engine, Li et al. rules) | Predicts legal moves | 0.01% illegal (Li et al. 2023) | < 1% illegal top-1 | — |
| othello-gpt | same; probes on resid_post of every layer, AdamW 6 epochs | Board is linear relative to the mover | mine/theirs ≈ 99.5% from L4; black/white ≈ 75% (Nanda et al. 2023) | best mine/theirs > 95% and ≥ 10 points above black/white, random-init and shuffled-label probes | random-init model; shuffled labels; absolute-color probe |
| othello-gpt | 150 flips that change the legal set; intervention at the best of layers 3–5 (mirror along the square's mine−theirs probe direction) | Editing the representation changes predicted moves | flip error 0.10 vs 2.72 null | error < 50% of null and < random-direction error (< null only = partial) | no intervention; random unit direction with same displacement |
| curve-detectors | Lucent InceptionV1 mixed3b neurons 379, 406, 385, 343, 342, 388, 324, 340, 330, 349; arcs r ∈ {15, 30, 60, 110}px × 72 orientations + straight lines | Orientation-tuned | radial tuning curves (Cammarata et al. 2020) | median selectivity (1 − circular variance) ≥ 20 random mixed3b neurons' median + 0.2 | 20 random non-curve neurons |
| curve-detectors | same | Prefer curves to lines | synthetic curves ≫ lines | ≥ 8/10 neurons: best curve > 1.5× best straight line (≥ 5 = partial) | straight lines |
| curve-detectors | same | Tile all orientations | joint tuning curves cover 360° | largest gap between preferred orientations ≤ 90° | — |

## Batch 3 — geometry (criteria written before any batch-3 run)

| Entry | Model / data | Claim | Published value (source) | Pass criterion | Controls / baselines |
|---|---|---|---|---|---|
| calendar-circles | GPT-2 small resid_pre L7; 16 templates × 7 days / 12 months; template means removed | Days and months are in cyclic order | circles in GPT-2 L7 (Engels et al. 2024, Fig. 1) | cyclic-order permutation p < 0.05 for both, and ≤ 10% of 40 random-word sets reach p < 0.05 (only one of days/months = partial) | random word sets of size 7 and 12 |
| calendar-circles | GPT-2 SAE `gpt2-small-res-jb` L7, top ≤ 24 latents by class-mean activation | SAE latents reconstruct the circle | Fig. 1 (SAE clusters) | cyclic order ≥ 0.7 for days and months | — |
| calendar-circles | Qwen3-1.7B, 42 prompts "Let's do some day of the week math. {N} days from {D} is"; layers 2–20 | The circle is causally used | circular-probe patching works in Mistral 7B / Llama 3 8B (§5) | some layer ≥ 50% success with random-plane control ≤ 20% at that layer (success > random + 20 points = partial). Layer selection across 7 layers is exploratory and reported in full | random 2-D plane, same displacement |
| number-helix | Pythia 160M/410M/1B/1.4B/2.8B; numbers 0–99 after 3 few-shot prefixes | Embeddings periodic with T = 2, 5, 10 | GPT-J Fourier peaks 2, 5, 10 (Kantamneni & Tegmark 2025); 2, 2.5, 5, 10 (Zhou et al. 2024) | ≥ 2 of {2, 5, 10} among the top-4 periods of Pythia-2.8B (±5%) | — |
| number-helix | same, 80/20 split over a | Numbers lie on a helix (T = 2, 5, 10, 100) | helix fits GPT-J / Pythia-6.9B / Llama-3.1-8B | Pythia-2.8B best-layer held-out R² ≥ 0.5 and ≥ 0.2 above a shuffled-a helix (≥ 0.1 = partial) | shuffled helix; linear-only; circles-only |
| number-helix | same; few-shot a+b= with a+b < 100, correct prompts only | Helix patching recovers most of the full-layer effect | helix 7.21 vs full 8.34 (86%), PCA 6.13 (GPT-J) | helix ≥ 50% of full-layer recovery and ≥ linear-only (> linear-only = partial) | full layer, PCA-9, linear-only, circles-only |
| linebreak-manifold | Qwen3-1.7B, 400 WikiText-103 paragraphs wrapped at 40–150 chars; GPT-2 small control | Count is linearly decodable | Haiku R² 0.985; Qwen3-8B 0.852 (T-Tech) | best held-out R² ≥ 0.7 and ≥ 0.5 above shuffled labels | shuffled labels; GPT-2 small |
| linebreak-manifold | same | Low-dimensional curved manifold | Haiku: 6 PCs = 95% | 6 PCs ≥ 80% of variance of per-count means | — |
| linebreak-manifold | same | Count present in the first layers | Haiku R² 0.985 after layer 1 (not replicated by T-Tech) | best R² in layers 0–2 ≥ 0.9 × best overall | — |
| linebreak-manifold | same | Model breaks lines like the text | Haiku newline prob 0.79 in example | P(newline) AUC for true breaks ≥ 0.9 | GPT-2 small |
| belief-fractal | 4-layer transformer (d 64, 1 head, LN) trained here on Mess3 (x = 0.05, α = 0.85), 30k steps × 512 seqs of 10 tokens | Beliefs linear in the final residual stream | MSE ≈ 4e-4 (Shai et al. 2024) | held-out R² ≥ 0.9 and ≥ 0.3 above random init (≥ 0.2 = partial) | random-init model |
| belief-fractal | same | Near-optimal prediction | trained near entropy rate | excess loss < 10% of the uniform-to-optimal gap | — |
| colour-bicone | FLUX.1-schnell VAE; 400 HSV-uniform solid colors, 128 px | Colors occupy a 3-D subspace | ≈ 100% in 3 PCs (Pach et al. 2026) | ≥ 95% | random-noise images |
| colour-bicone | same | HSL-like axes | PC1 = lightness, angle = hue, radius = saturation | corr(PC1, L), circular corr(angle, hue), corr(radius, chroma) all ≥ 0.8 (first two only = partial) | — |

## Later batches (criteria are added here before each batch runs)

| Batch | Entries | Models | Where |
|---|---|---|---|
| 2 — circuits | copy-suppression, greater-than, othello-gpt, curve-detectors (+BSF block) | GPT-2 small, TL othello-gpt, Lucent InceptionV1 | local |
| 3 — geometry | calendar-circles, number-helix, linebreak-manifold, belief-fractal, colour-bicone, toy-superposition | GPT-2 small, Pythia 160M–2.8B, Qwen3-1.7B, tiny models, FLUX VAE | local |
| 4 — Neural Geometry series | world-inside (mountain car), manifold-steering, geometric-calculator (probes), sae-manifolds (synthetic), stories-in-space, block-sparse-featurizers | own tiny models, Qwen3-1.7B, Qwen2.5-1.5B-Instruct, Pythia | local |
| 5 — features / steering / lenses | sae-features, actadd, refusal-direction, emotion-concepts, jlens, neuron-basis | GPT-2 small/XL, Qwen2.5-1.5B-Instruct, Qwen3-0.6B/1.7B | local |
| 6 — attribution graphs | attribution-graph (Qwen3-1.7B circuit-tracer; Gemma-2-2B via Neuronpedia API) | Qwen3-1.7B, Gemma-2-2B | local + Neuronpedia |
| 7 — paper scale | calculator + manifold steering + Engels interventions (Llama-3.1-8B or Qwen3-8B/Mistral-7B), helix (Pythia-6.9B/GPT-J), emotions/J-lens/linebreak at 8B (Qwen3-8B), MoE neurons (OLMoE-1B-7B) | 6–8B models | RunPod, ≤ $20 total |
