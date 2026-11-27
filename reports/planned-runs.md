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

## Batches 4–5 — Neural Geometry series, features, steering, lenses, neurons, graphs (criteria written before running)

| Entry | Model / data | Claim (published) | Pass criterion (partial) | Controls |
|---|---|---|---|---|
| sae-manifolds | TopK SAEs (512 atoms) on 16 synthetic manifolds in d = 128, L0 = 4, k ∈ {3…25} | capture peaks near k = L0 = 4 (Bhalla et al. 2026) | capture fraction peaks at k ∈ {3, 4, 6} | — |
| sae-manifolds | same | larger k shatters manifolds | shattering at max k ≥ 30 pts above min k (any increase) | — |
| block-sparse-featurizers | 32 sphere concepts in 3-D subspaces of d = 128, L0 = 4; BSF (64 blocks × 4, top-4) vs TopK SAE (256 latents, k = 16) | BSF 0.93–0.97 vs TopK 0.53 per-block R² (Fel et al. 2026) | BSF median ≥ 0.8 and ≥ 0.2 above SAE (BSF > SAE) | TopK SAE |
| block-sparse-featurizers | InceptionV1 mixed3b, 288 curve stimuli; top-16 PCs | harmonics 1–3 explain 89% | ≥ 50% and > shuffled-orientation null p95 (> null) | shuffled orientations |
| attribution-graphs | 9 public Gemma-2-2B Neuronpedia graphs | two-hop path Dallas → Texas → Austin (Lindsey et al. 2025) | path in all 3 capital graphs (in ≥ 1) | — |
| attribution-graphs | en/fr/zh antonym graphs | shared middle-layer features | middle-layer Jaccard > early and late | — |
| neuron-basis | Qwen3-1.7B, 48 × 512 WikiText-103 tokens | 92.2% ablatable: +2.9% loss, KL 0.14 (Goodfire, MoE); KL ≈ 0.80 (ericzhou12, Qwen3-1.7B) | top-480 rel. loss < 25%, KL < 1.0, random-480 ≥ 3× worse (random only) | random-480, drop-top-480, band P25–75 |
| neuron-basis | same | dropping only the top activations is catastrophic | loss increase ≥ 5× that of keeping top-480 | — |
| neuron-basis | same, grad×act top-1% cells | gate-open quadrants carry causal weight; 71–72% gate-closed | ≥ 80% top cells Q1+Q2 and ≥ 50% of activations Q3+Q4 (≥ 60% top cells) | — |
| refusal-direction | Qwen2.5-1.5B-Instruct; 128 train / 32 val / 64 test prompts each (Arditi et al. splits) | ablation removes refusal; addition induces it | ablation −50 pts on harmful with random < 15 pts change (−25); addition +50 pts on harmless with random < 15 (+25) | norm-matched random direction |
| actadd | GPT-2-XL; weddings vector (layer 20), Love−Hate (layer 6); c ∈ {0,1,2,4,8}; 8 samples × 3 prompts | > 90% on-topic weddings (Turner et al.) | c = 4 on-topic ≥ 50% and ≥ 30 pts above unsteered and random (≥ 15) | random vector |
| actadd | same | Love−Hate makes text positive | ≥ 20 pts positive above baseline and random | random vector |
| jlens | Qwen3-1.7B + pre-fitted Neuronpedia J-lens; 4 prompts with unspoken intermediates | J-lens surfaces intermediates better than logit lens (Gurnee et al. 2026) | lower median best-rank than logit lens and ≥ half in top-10 (lower median) | logit lens |
| jlens | 8 country-swap prompts, layers 30–70% depth | swapping J-lens coordinates swaps the concept (59% top-5) | ≥ 50% top-5 and ≥ 30 pts above random swap (≥ 20) | random direction swap |
| emotions | Qwen3-1.7B, 80 emotions × 12 topics × 2 self-written stories; Warriner norms | PC1 = valence (r 0.81) | |r| ≥ 0.6 and > shuffled-label p95 (> p95) | shuffled labels |
| emotions | same | PC2 = arousal (r 0.66) | |r| ≥ 0.4 (≥ 0.25) | — |
| emotions | Tylenol dose sweep 250–16,000 mg | afraid ↑, calm ↓ with dose | afraid slope > 0 and calm slope < 0 | — |
| emotions | 120 activity pairs, steering 0.5 × residual norm on option A | blissful +212 Elo, hostile −303 | blissful > base > hostile by ≥ 0.05 each, random within 0.05 (ordering only) | random vector |
| manifold-steering | Qwen3-1.7B weekday arithmetic, layers 12/16/20, 64-d PCA, periodic spline | E_BC 0.34 vs 0.93 (Wurgaft et al. 2026) | manifold E_BC < 0.7 × linear (< linear) | linear chord |
| manifold-steering | same | manifold paths visit intermediate days | ≥ linear + 15 pts (> linear) | linear chord |
| calculator-probes | Pythia-410M/1.4B/2.8B, 1,500 "{a}+{b}=" prompts, ridge probes T = 2…150 | base-10 periods prominent (Feucht et al. 2026) | ≥ 3 base-10 periods in top 8 and R²(T=10) ≥ neighbours + 0.1 (≥ 2 in top 8) | neighbouring periods |
| world-model | mountain-car world model trained here (100 rollouts, 64×64) | position R² ≈ 0.95 | position R² ≥ 0.8 and ≥ 0.2 above untrained encoder | untrained encoder |
| world-model | same | arc r 0.996 vs chord 0.06 | manifold r ≥ 0.9 and > chord (> chord) | straight chord |
| stories-in-space | Qwen2.5-1.5B-Instruct, 160 SimpleStories, 6 emotions | probes predict ratings (RMSE 0.09) | held-out R² ≥ 0.5 (≥ 0.3) | — |
| stories-in-space | same | behaviour–activation distance r = .92 | r ≥ 0.6 (≥ 0.3) | — |
| sae-features | GPT-2 small res-jb L8; 5 concept features; α ∈ {0…80} | Golden Gate clamping (Templeton et al. 2024) | Golden Gate on-topic ≥ 50% and ≥ 30 pts above base and random (≥ 15) | random direction |
| sae-features | same | steering generalizes | ≥ 3/5 concepts gain ≥ 30 pts (≥ 2) | random direction |

## Grokking progress measures from the companion runs (added 2026-11-27, before running)

Not a blind test: the companion project preregistered these hypotheses (its `docs/preregistration.md`, tag `prereg-v1`)
and we had read its verdict table (11/11 seeds for H1 and H2) before writing this. Our module re-implements its published
definitions (`docs/operationalizations.md`) independently, from its raw per-checkpoint metrics (`analysis/ckpt_metrics.npz`,
`analysis/phases.json` on Hugging Face), as a code-level cross-check. The companion's H3–H5 verdicts are quoted from its
`reports/tables/hypotheses.json` and are not re-tested here.

| Entry | Data | Claim (Nanda et al. 2023, §5) | Our metric (companion definitions) | Pass criterion |
|---|---|---|---|---|
| grokking-runs | 11 wd = 1 seeds, 401 checkpoints each | Restricted loss falls before test loss | first checkpoint ≥ t1 where the restricted loss (all pairs) is ≤ 10% of its value at t1, vs the same for test loss | earlier in ≥ 10 of 11 seeds (companion rule: fewer than two failures) |
| grokking-runs | same | Excluded loss rises during circuit formation | excluded loss (train pairs, key frequencies removed) at t2 ÷ at t1 | > 10× in ≥ 10 of 11 seeds |

## Amendments (dated; each made after a smoke test and before the corresponding full run)

| Date | Entry | Amendment | Reason |
|---|---|---|---|
| 2026-11-27 | linebreak-manifold, stories-in-space | Method fix, criteria unchanged. Ridge strength is chosen per layer from {1, 10, …, 10⁵} on held-out *training* documents (grouped), instead of a fixed α. The linebreak analysis drops position 0 of each document. | Smoke test: fixed α = 10 on 2,048 standardized features overfit (negative held-out R² in late layers). The first token's massive activations (attention sink) dominated low-count means (3 PCs = 99.9% of variance). |
| 2026-11-27 | colour-bicone | Weights from `diffusers/FLUX.1-vae` (ungated, same FLUX.1 VAE config) instead of the gated `black-forest-labs/FLUX.1-schnell`. | The official repository requires accepting terms on the user's account. |
| 2026-11-27 | jlens | (1) The arithmetic prompt leaves the intermediate-rank metric (still shown in the grids). (2) A paper-style swap is added as a second row with the same thresholds: exchange coefficients in a sparse non-negative decomposition over the top-25 J-lens vectors. The pre-stated simple coordinate swap stays and is reported as is. | (1) Qwen3 tokenizes numbers digit by digit: "21" → "2","1", and "42" and "49" share the token "4", so the rank of a multi-digit intermediate is undefined. (2) The quick run's simple swap (exchanging projections on two non-orthogonal unit directions) is a weaker edit than the paper's J-space coefficient swap. Both are reported to avoid choosing a method by its result. |
| 2026-11-27 | actadd | GPT-2 XL loaded in bfloat16; samples batched with the KV cache; steering added on the prompt pass only. | fp32 filled the 8 GB GPU (54 min smoke test). Greedy generation with the new hook was verified token-identical to recomputing every step. |
| 2026-11-27 | calendar-circles (causal test), manifold-steering | Day-arithmetic prompts changed to templates the model solves ("If today is D, then the day after / two days later / three days later it is", 3 prefixes). Only correctly answered prompts define centroids, steering sources and patches. manifold-steering steers in the 6-d span of the 7 answer-day centroids instead of a 64-d PCA. Criteria unchanged. | CPU check: Qwen3-1.7B answered Engels et al.'s "N days from D" template correctly 0–57% of the time per N and said "Wednesday" for 18 of 42 prompts. Centroids of the true answer therefore did not reflect the model's representation, and both steering paths collapsed to the same output. A 64-d PCA on ~60 prompts spans almost all prompt variance, so overwriting it creates off-distribution hybrids. |
| 2026-11-27 | calculator-probes | (1) Ridge strength chosen per period on training rows (as `RidgeCV(alpha_per_target=True)`) instead of a fixed α = 10. (2) A second row ranks periods by peak score R²(T) − mean R²(T±1), same thresholds. The pre-stated global-rank row stays. | (1) Fixed α overfit (held-out R² at T = 10 was −0.78); one α chosen by the mean over all periods picks maximal shrinkage. Synthetic test: a planted T = 10 circle now gives R² ≈ 1. (2) Periods longer than the range of a+b act as a monotone magnitude code and win a global ranking (smoke test: top 8 = 143–150), which says nothing about base-10 peaks. |
| 2026-11-27 | manifold-steering | Steering layers 12/16/20 → 22/24/26 (79–93% depth; the paper uses 28 of 32 = 87.5%). | Endpoint check, independent of the manifold-vs-linear comparison: writing day j's centroid at the last token made the model answer j in 0–8% of cases at layers 12–20 (even when the whole residual is replaced: the answer is not formed there yet) and 92–100% at layers 22–26. The E_BC definition and both criteria are unchanged. |
| 2026-11-27 | world-model | Probe train/test split by rollout instead of by frame; extra (non-verdict) velocity probe on the GRU state; timing logs. | Neighbouring frames of one rollout are near-duplicates, so a per-frame split leaks; single frames cannot show velocity (frame-latent velocity R² 0.03 in the smoke test). |
| 2026-11-27 | stories-in-space | Ratings on a 0–9 scale (single-digit answers) instead of 0–10. All prefixes of a story are batched, with position ids from the attention mask. | Qwen tokenizes "10" as "1","0", so the 0–10 readout folded P(10) into P(1). Batching makes the full run feasible (the smoke test took 32 min for 20 stories, one forward pass per prefix). Criteria unchanged. |
| 2026-11-27 | sae-features | Features chosen by their weakest peak activation over three probe texts per concept (relative to WikiText) instead of one focus token. Steering evaluation and criteria unchanged. | One focus token picked word-sense features: French " pain" (bread) → an English "pain/killers" feature; " Hogwarts" → a prison/institution feature. |
| 2026-11-27 | number-helix | The patching claim needs ≥ 20 correctly solved prompts in the largest model; otherwise the verdict is **n/a**, not pass/fail. The criterion itself is unchanged. | In the `--quick` smoke test Pythia-410m solved 1 of 59 few-shot `a+b=` prompts (CPU spot check: 23+45 → "53"), so a "pass" on n = 1 would be meaningless. Tokenization was checked: NeoX has single tokens for 0–99. |

## Later batches (criteria are added here before each batch runs)

| Batch | Entries | Models | Where |
|---|---|---|---|
| 2 — circuits | copy-suppression, greater-than, othello-gpt, curve-detectors (+BSF block) | GPT-2 small, TL othello-gpt, Lucent InceptionV1 | local |
| 3 — geometry | calendar-circles, number-helix, linebreak-manifold, belief-fractal, colour-bicone, toy-superposition | GPT-2 small, Pythia 160M–2.8B, Qwen3-1.7B, tiny models, FLUX VAE | local |
| 4 — Neural Geometry series | world-inside (mountain car), manifold-steering, geometric-calculator (probes), sae-manifolds (synthetic), stories-in-space, block-sparse-featurizers | own tiny models, Qwen3-1.7B, Qwen2.5-1.5B-Instruct, Pythia | local |
| 5 — features / steering / lenses | sae-features, actadd, refusal-direction, emotion-concepts, jlens, neuron-basis | GPT-2 small/XL, Qwen2.5-1.5B-Instruct, Qwen3-0.6B/1.7B | local |
| 6 — attribution graphs | attribution-graph (Qwen3-1.7B circuit-tracer; Gemma-2-2B via Neuronpedia API) | Qwen3-1.7B, Gemma-2-2B | local + Neuronpedia |
| 7 — paper scale | calculator + manifold steering + Engels interventions (Llama-3.1-8B or Qwen3-8B/Mistral-7B), helix (Pythia-6.9B/GPT-J), emotions/J-lens/linebreak at 8B (Qwen3-8B), MoE neurons (OLMoE-1B-7B) | 6–8B models | RunPod, ≤ $20 total |
