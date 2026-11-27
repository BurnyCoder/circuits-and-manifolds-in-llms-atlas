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

## Later batches (criteria are added here before each batch runs)

| Batch | Entries | Models | Where |
|---|---|---|---|
| 2 — circuits | copy-suppression, greater-than, othello-gpt, curve-detectors (+BSF block) | GPT-2 small, TL othello-gpt, Lucent InceptionV1 | local |
| 3 — geometry | calendar-circles, number-helix, linebreak-manifold, belief-fractal, colour-bicone, toy-superposition | GPT-2 small, Pythia 160M–2.8B, Qwen3-1.7B, tiny models, FLUX VAE | local |
| 4 — Neural Geometry series | world-inside (mountain car), manifold-steering, geometric-calculator (probes), sae-manifolds (synthetic), stories-in-space, block-sparse-featurizers | own tiny models, Qwen3-1.7B, Qwen2.5-1.5B-Instruct, Pythia | local |
| 5 — features / steering / lenses | sae-features, actadd, refusal-direction, emotion-concepts, jlens, neuron-basis | GPT-2 small/XL, Qwen2.5-1.5B-Instruct, Qwen3-0.6B/1.7B | local |
| 6 — attribution graphs | attribution-graph (Qwen3-1.7B circuit-tracer; Gemma-2-2B via Neuronpedia API) | Qwen3-1.7B, Gemma-2-2B | local + Neuronpedia |
| 7 — paper scale | calculator + manifold steering + Engels interventions (Llama-3.1-8B or Qwen3-8B/Mistral-7B), helix (Pythia-6.9B/GPT-J), emotions/J-lens/linebreak at 8B (Qwen3-8B), MoE neurons (OLMoE-1B-7B) | 6–8B models | RunPod, ≤ $20 total |
