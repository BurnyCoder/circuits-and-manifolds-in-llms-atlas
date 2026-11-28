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

## Amendments (dated; meant to be made after a smoke test and before the full run; exceptions in the next section)

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
| 2026-11-27 | stories-in-space | "Behaviour and activation geometry agree" is computed as RSA over up to 600 prefixes (Spearman correlation of pairwise distances), with a story-block shuffle null, instead of over 3–6 emotion centroids. Same threshold r ≥ 0.6. | The smoke test had only 3 emotions with enough dominant prefixes, so the correlation rested on 3 distance pairs: statistically meaningless whatever its sign. |
| 2026-11-27 | sae-features | Each concept's feature is the top match of Neuronpedia's explanation search over this SAE's auto-interp labels (cached in `results/sae-features/neuronpedia_search.json`); activation on probe texts is only the fallback. Steering evaluation and criteria unchanged. | Activation-based selection picked wrong features. With one focus token, French " pain" (bread) gave an English "pain/killers" feature and " Hogwarts" a prison feature. With three probes, the bridge feature lost to a generic place-name feature. Auto-interp labels come from top-activating examples, independent of the steering test. All three variants are recorded in learnings.md. |
| 2026-11-27 | sae-manifolds | Metrics re-implemented per the paper's App. E: in-distribution (superposed) evaluation rows, greedy decoder-direction restricted R², 10%/30-row support with per-atom magnitude floor; 48 instances (6 variants per type) as in the paper. Claim 1 is judged on restricted R² (the paper's capture score), same k ∈ {3, 4, 6}. Claim 2 is restated as small k → shattering, large k → dilution. | The first full run gave "capture 0% at every k", and claim 1 still printed "pass" because the argmax of all-zero fractions is the first k (a verdict bug). Our evaluation showed each manifold alone, where TopK always fires k atoms, and used 16 instances, so other manifolds' atoms crossed the 10% support threshold. The pre-stated claim 2 ("larger k shatters") had the paper's direction reversed (Fig. 4B: low k shatters, high k dilutes). |
| 2026-11-27 | number-helix | The patching claim needs ≥ 20 correctly solved prompts in the largest model; otherwise the verdict is **n/a**, not pass/fail. The criterion itself is unchanged. | In the `--quick` smoke test Pythia-410m solved 1 of 59 few-shot `a+b=` prompts (CPU spot check: 23+45 → "53"), so a "pass" on n = 1 would be meaningless. Tokenization was checked: NeoX has single tokens for 0–99. |

## Post-review disclosures and control fixes (2026-11-28)

An adversarial methods review ([`reviews/methods-review.md`](reviews/methods-review.md)) matched run logs, `quick`
flags and commit times. It found that the rule in the heading above was broken several times. Those rows are now
**post-hoc**: they stay on the site and in every table, marked with the reason, and are excluded from the pre-stated
tally. The list lives in one file, [`src/atlas/common/post_hoc.py`](../src/atlas/common/post_hoc.py), and is applied by
`common.io.write_entry`.

| Entry | What happened | Status |
|---|---|---|
| attribution-graphs | Three full runs (19:00, 19:01, 19:02 CET, 2026-11-27) moved the two-hop result from 0/3 to 1/3 to 3/3 while the analysis was edited; the criteria were committed 11 minutes after the passing run. The multilingual criterion was also written after the runs. No negative control. | both rows post-hoc |
| sae-manifolds | Claim 1's metric (restricted R² peak) was adopted after a full run whose log already showed the peak at k = 6. Claim 2 was restated (direction corrected) after the same run. | both rows post-hoc |
| manifold-steering | Layers 12/16/20 → 22/24/26 after two failing full runs; the second run started before the amendment was committed. The endpoint check that motivated it was not in the code (now it is, and its per-layer output is saved). | both rows post-hoc |
| sae-features | The feature-selection rule changed twice after smoke tests that logged steering rates; only the final rule passes claim 2. | both rows post-hoc |
| jlens | The arithmetic prompt left the rank metric after its ranks had been seen: first-digit ranks 1360/1088 (J-lens) vs 16/42 (logit lens), i.e. against the J-lens. With them included the row would still pass (medians 6 vs 14, top-10 3/5 vs 2/5, recomputed by the review). The paper-style swap was added after the pre-stated swap failed. | both rows post-hoc |
| calculator-probes | The peak-score row was added after the pre-stated ranking failed. | row 2 post-hoc |
| stories-in-space | RSA replaced the centroid analysis after the smoke test. | row 2 post-hoc |
| calendar-circles | The day-arithmetic templates changed after the smoke test showed the causal test failing. | causal row post-hoc |
| grokking-runs | Rows 1–3 never had a criterion here; the companion project's results were known. | rows 1–3 descriptive (post-hoc) |
| several | Some amendments were committed minutes after the runs they enabled had started (`c57cb29` 21:15 vs a run at 21:12; `f54feef` 22:44 vs a run at 22:25). | disclosed |

**Control fixes.** Committed in `49014a1` (01:05 CET, 2026-11-28) before the reruns that produced the current numbers.
Fixes of existing controls keep their criteria; new control rows were written before they ran, but after the main
results were known, so they are listed as post-hoc as well.

| Entry | Fix | Criterion |
|---|---|---|
| induction-heads, othello-gpt | The "random-init" models were all-zero networks: TransformerLens pretrained configs carry `init_weights=False`. The config is now copied with `init_weights = True`. | unchanged |
| calendar-circles | The random-plane control is norm-matched to the circle-plane displacement (it was 1–2 orders of magnitude smaller). | unchanged |
| refusal-direction | New control: a difference-of-means direction from randomly relabelled training prompts, same layer and norm, ablated on harmful and added on harmless prompts. | both refusal-rate changes < 15 points |
| world-model | New control: the same spline isometry on the untrained encoder's latents. | trained (r_arc − r_chord) exceeds untrained by ≥ 0.2 |
| block-sparse-featurizers | New control: orientation-harmonic share of 10 random non-curve mixed3b channels vs the 10 curve neurons. | curve share − random share ≥ 10 points |
| manifold-steering | Endpoint check in code for layers 12, 16, 20 (reported only) and 22, 24, 26. | reported, no verdict |
| sae-features | Golden Gate specificity counted separately ("golden gate" or "san francisco" only); claim renamed to a bridge feature. | reported, no verdict |

## J-lens confirmatory test on fresh prompts (added 2026-11-28 01:35 WET, before running)

The methods review (M11) found the J-lens intermediate-rank row rests on n = 3 and is post-hoc. This test reuses the
pre-stated metric unchanged on 24 new two-hop prompts written before any lens readout was computed on them. The prompts
were filtered only by tokenization (every intermediate is one Qwen3 token with a leading space; checked with the
tokenizer alone).

| Entry | Data | Claim | Metric | Pass criterion |
|---|---|---|---|---|
| jlens | 24 prompts, 3 templates × 8: "Fact: The capital of the state containing {city} is" (Miami, Chicago, Seattle, Detroit, Las Vegas, Philadelphia, Cleveland, New Orleans); "Fact: The capital of the country where the {landmark} is located is" (Colosseum, Kremlin, Acropolis, Taj Mahal, Great Wall, Sagrada Familia, pyramids of Giza, CN Tower); "Fact: The language spoken in the country whose capital is {capital} is" (Berlin, Lisbon, Tokyo, Moscow, Stockholm, Warsaw, Oslo, Ankara) | The J-lens surfaces the unspoken intermediate (state / country) better than the logit lens | best rank over lens layers of the intermediate token at the last position, J-lens vs logit lens, per prompt | the J-lens rank is strictly better on more prompts than the reverse **and** a two-sided sign test over non-tied prompts gives p < 0.05; partial if it wins more prompts with p ≥ 0.05; fail otherwise |

## Review follow-ups: new pre-stated tests and control fixes (added 2026-11-28 04:40 UTC, before running)

These close the open items of the methods review ([`reviews/methods-review.md`](reviews/methods-review.md); status in
[`reviews/response-to-review.md`](reviews/response-to-review.md)). Everything below was written before any of its
numbers were computed. Prompts, word lists and labels were checked with the tokenizer only.

**Rules.**
- *New pre-stated tests* add verdict rows. They count in the pre-stated tally, including the ones that re-test claims
  whose earlier rows are post-hoc. Earlier rows stay as they are, with their post-hoc flags.
- *Control fixes* replace a weak control or add uncertainty to an existing row. They keep that row's criterion.
- *CI rule* (rows marked "CI rule"): pass if the whole 95% CI meets the threshold, fail if the whole CI misses it,
  otherwise **partial (inconclusive)**.
- Bootstrap CIs use 2,000 percentile resamples over the stated unit (prompt, path, document, story, seed × instance).
  Rate CIs are Wilson 95% intervals (`common.stats.wilson_ci`).
- Every run starts from a committed tree, and `src_dirty_at_start` is recorded.

### New pre-stated tests

| Entry (review ID) | Data | Claim | Metric | Pass criterion |
|---|---|---|---|---|
| attribution-graphs (C1) | The 3 public Neuronpedia two-hop graphs, aggregated by the committed `summarize` and scored by `path_strength` (both unchanged). Wrong intermediates, fixed now: Dallas graph "California", "Florida", "China"; Oakland graph "Texas", "Florida", "China"; Shanghai graph "Japan", "Texas", "California". | The two-hop path runs through the *right* intermediate, not through any labelled node | (a) path strength with the correct label. (b) Path found (> 0) for each of the 9 wrong labels. (c) Label-permutation null: 1,000 random permutations of the labels of all non-embedding, non-logit groups; p_g = fraction of permutations whose correct-label path strength ≥ observed. | **pass**: in all 3 graphs the correct path is > 0 with p_g < 0.05, **and** wrong-label paths are found in ≤ 1 of 9 checks; **partial**: p_g < 0.05 in 1–2 graphs; **fail** otherwise |
| attribution-graphs (C1) | The 3 antonym graphs (en/fr/zh), plus a control triplet of unrelated graphs: `gemma-fact-dallas-austin`, `gemma-addition`, `gemma-michael-jordan` | Language-independent features concentrate in the middle layers | Jaccard of transcoder-feature sets per layer band (code unchanged), for the multilingual and the control triplet | **pass**: the multilingual middle-band Jaccard is above the multilingual early and late bands, **and** ≥ the control triplet's middle-band Jaccard + 0.05; **partial**: only one of the two holds; **fail** otherwise |
| sae-manifolds (C2) | The same 48 instances, k grid and 3,000 steps. 6 new SAE seeds per k (training seed `ctx.seed + 1000·(s+1) + k`, evaluation seed `+ 500`), s = 0…5. | Subspace capture has an interior sparsity optimum | Mean restricted R² per k. k* = argmax over seeds 0–2. On held-out seeds 3–5: Δ = R²(k*) − max(R²(3), R²(25)), with a 95% CI from a two-level bootstrap (seeds, then instances). The originally pre-stated capture-fraction peak is also reported. | **pass**: k* ∈ {4, 6} and the CI of Δ > 0; **partial**: k* ∈ {8, …, 20} and the CI of Δ > 0; **fail** otherwise (including k* ∈ {3, 25}) |
| sae-manifolds (C2) | Held-out seeds 3–5 of the run above | Low k shatters, high k dilutes | dilution(25) − dilution(3) and shattering(3) − shattering(25), as fractions of the 48 instances, averaged over seeds; CI by the same bootstrap | **pass**: the dilution rise is ≥ 0.30 with its CI > 0, **and** the shattering drop is > 0; **partial**: both point estimates > 0; **fail** otherwise |
| manifold-steering (M2) | Layers 22, 24 and 26, all three (no layer selection). Up to 4 correctly answered source prompts per start day (the first 4 in prompt order). Every ordered pair 2 or 3 days apart. | Steering along the smooth manifold stays closer to the behaviour manifold than a path that visits the same centroids | Third path type: piecewise-linear through the same day centroids along the shorter direction, K = 13 points, reaching each intermediate centroid at the same t as the spline. Per path, Δ = E_BC(spline) − E_BC(piecewise-linear) (`bc_dist` unchanged), averaged over paths; 95% CI by bootstrap over source prompts. | **pass**: the CI of Δ lies below 0 at all 3 layers; **partial**: at 1–2 layers; **fail**: at none |
| calculator-probes (M6) | 1,500 "{a}+{b}=" prompts (same sampling). New split: 80/20 by unordered pair {a, b}, so no pair or its commuted form is on both sides. Fixed layer ⌊2·n_layers/3⌋ (Pythia-2.8B: 21; also reported for 410M and 1.4B at their ⌊2n/3⌋). | Base-10 periods stand out among the Fourier probes | Peak score as before, but a period counts as a peak only if R²(T) > 0.05. Hits = base-10 periods among the top-8 valid peaks. Null 1: 200 permutations of s across prompts (same pipeline) → p_perm. Null 2: 10,000 random 6-period sets from 2–150, counted against the same valid top-8 → p_set. | **pass** (Pythia-2.8B): hits ≥ 3 with p_perm < 0.05 and p_set < 0.05; **partial**: hits = 2 with both p < 0.05; **fail** otherwise |
| number-helix (M7) | Same prompts and patch layers as the main run | Patching along the helix beats generic 9-D subspaces | At every patch layer whose full-layer median recovery is ≥ 0.5: per-prompt recovery for the helix patch, PCA-9 (unchanged), and a random 9-D subspace of the 99-D span of the number-token activations at that layer (mean of 5 draws; the actual activation difference projected onto it). Paired differences helix − control, bootstrap 95% CI of the mean over prompts. | **pass**: at every qualifying layer the CI is > 0 against both random-9 and PCA-9; **partial**: > 0 against random-9 only; **fail** otherwise; **n/a**: no qualifying layer or < 20 solved prompts |
| emotions (M12) | 8 templates × 7 doses (250 … 16,000), slot {X} = "{d} mg of Tylenol" or, as the benign control, "{d} mg of sugar". Templates: "I just had {X}. Should I be worried?" · "My son just swallowed {X}. What should I do?" · "I had {X} this morning. Is that okay?" · "Yesterday I had {X} before bed. Is that a problem?" · "My friend says she just had {X}. Should we call someone?" · "Is it fine that I've had {X} today?" · "Over the last hour I have had {X}. Anything I should know?" · "I accidentally had {X} at once. Is that safe?" | Emotion vectors track danger, not the size of the number | Per template: slope over log10(dose) of the last-token projection on 'afraid' minus the projection on 'calm' (same vectors and layer). S = mean slope over templates. Null: 10,000 permutations of dose order, independently per template (and per condition for the difference). | **pass**: S_Tylenol > 0 with p < 0.05 **and** S_Tylenol − S_sugar > 0 with p < 0.05; **partial**: only the first; **fail** otherwise |
| block-sparse-featurizers (M13) | The toy as before, with 3 training seeds; evaluated on superposed inputs (L0 = 4 concepts active, as in training) | Blocks capture concepts that a TopK SAE fragments, in distribution | For concept c, the rows where c is active and its true component. BSF: the block with the largest mean norm on those rows, decoded alone. TopK: the 4 atoms with the largest mean activation on those rows. R² against the component; median over concepts, then mean over seeds. | **pass**: BSF ≥ 0.8 and ≥ TopK + 0.2, with BSF > TopK in all 3 seeds; **partial**: BSF > TopK on average; **fail** otherwise |
| block-sparse-featurizers (M13) | `lucent` `inceptionv1(pretrained=False)` (parameter std > 0 asserted); same stimuli and analysis | The orientation ring is learned, not a property of any conv net on these stimuli | Harmonic 1–3 share of the top-16 PCs, trained vs random-init. Reported without a verdict: a per-channel circular-shift null (each channel's tuning shifted by a random orientation; the same share is expected up to PCA truncation, which shows the measure tracks per-channel smoothness). | **pass**: trained share ≥ random-init share + 10 points; **fail** otherwise |
| linebreak-manifold (M9) | Fresh documents: 400 WikiText-103 paragraphs drawn from those *not* used by the main run (seed `ctx.seed + 1`); same models and widths | Character count is linearly decodable in Qwen3-1.7B | Layer chosen on validation documents (every 5th training document); test R² at that layer, with a 95% CI by bootstrap over test documents; shuffled-label R² at the same layer | **CI rule** with threshold 0.7 and point − shuffled ≥ 0.5 required for pass |
| linebreak-manifold (M9) | Same fresh run | The count is already represented in the first layers | Ratio of test R² at the best validation layer among 0–2 to test R² at the validation-chosen layer; CI by bootstrap over test documents | **CI rule** with threshold 0.9 |
| stories-in-space (M9) | Fresh stories: the next 200 qualifying SimpleStories test stories after those used by the main run; same model, prompts and layers | The running emotional reading is linearly decodable | Layer chosen on validation stories (every 5th training story); test R² with a CI by bootstrap over test stories. Reported: a bag-of-words baseline (TF-IDF unigrams + bigrams of the prefix text, ridge α chosen on the same validation split). | **CI rule** with threshold 0.5 |
| stories-in-space (M9) | Same fresh run | Behaviour and activation geometry agree | RSA Spearman r at the validation-chosen layer (procedure unchanged, including the story-shuffle null) | **pass**: r ≥ 0.6 and above the null's 95th percentile; **partial**: r ≥ 0.3 and above the null's 95th percentile; **fail** otherwise |
| refusal-direction (M14) | The ablated harmful completions and the harmless baseline completions of the main run | Removing refusal leaves fluent text (not degeneration) | Per completion: mean per-token NLL of tokens 6–40 under the *unmodified* model (conditioned on the prompt), and the distinct-3-gram ratio. Numbers only; no text is stored. | **pass**: median NLL(ablated) ≤ median NLL(harmless baseline) + 1.0 nat **and** median distinct-3(ablated) ≥ 0.8 × harmless baseline; **fail** otherwise |
| ioi (m1) | Same prompts and p_ABC means | The circuit is faithful with the paper's (head, position) nodes | Keep circuit heads only at their class positions (Wang et al. §2.1): previous-token at S1+1; duplicate-token and induction at S2; S-inhibition, name movers, negative and backup name movers at END. Every other (head, position) is mean-ablated. Metric: fraction of the full logit difference, with a bootstrap CI over prompts. | **pass**: 60–120%; **partial**: 40–60%; **fail** otherwise |
| copy-suppression (m5) | The 8 existing prompts plus 16 written now (copied token in brackets): "I went to the bank, and after the bank I went to the" [bank] · "Alice gave Bob a book, and then Bob gave the book back to" [Bob] · "The river flows into the sea, and the sea meets the" [sea] · "We ate pizza on Friday and pizza again on" [pizza] · "London is in England. Rome is in Italy. Dublin is in" [England] · "My favourite colour is blue, and my sister's favourite colour is" [blue] · "He opened the door, walked through the door, and closed the" [door] · "Cats chase mice, dogs chase cats, and mice chase" [mice] · "The teacher asked the student, and the student asked the" [student] · "First comes winter, then spring, then summer, then" [spring] · "She sang a song, then another song, and finally a" [song] · "Coffee in the morning, coffee at noon, and tea in the" [morning] · "The king and the queen met the" [king] · "Water is wet and fire is" [wet] · "Up is the opposite of down, and left is the opposite of" [down] · "He likes dogs more than cats, but she likes cats more than" [cats] | L10H7 specifically suppresses the copied token | Per prompt, the control token = the most probable next token (clean run) that does not occur in the prompt. L10H7's direct effect on the copied token vs on the control token. | **pass**: the copied token's direct effect is lower than the control token's on ≥ 75% of the 24 prompts, **and** negative on ≥ 75%; **partial**: one of the two; **fail** otherwise |
| colour-bicone (m6) | Same 400 colours. Random-init FLUX VAE: `AutoencoderKL.from_config` of the same config, default initialisation, seeded. Reported without a verdict: PCA of the RGB values themselves (a linear-RGB baseline). | The HSL bicone is learned, not a property of any VAE on solid colours | min(r_light, r_hue, r_sat), trained vs random-init | **pass**: trained ≥ random-init + 0.2; **fail** otherwise |

### Control fixes (criteria unchanged)

| Entry (review ID) | Fix |
|---|---|
| induction-heads (m4) | The control ablates every set of non-induction layer-1 heads of the same size (up to 20 sets), instead of one random draw. The criterion uses the mean increase; the max is reported. |
| calendar-circles (m3) | Random-word pool of 93 single-token nouns (was 20), 40 sets. `cyclic_order_score` returns NaN when angles tie (counted as not significant). Reported: share of class-mean variance in PC1–2, days/months vs random sets. |
| curve-detectors (m2) | Control median also reported over responsive control neurons only (max response > 0); bootstrap CI of the curve − control difference over neurons. |
| number-helix (M7) | Row 2: 5-fold CV over a; held-out R² is the max over layers for every basis, the shuffled control included. Row 3: a pass also requires full-layer recovery ≥ 0.5 at the reported layer (m7). |
| othello-gpt (M8) | Occupied-square accuracy in the row text; bootstrap CI over test games for the best-layer accuracy. |
| sae-features (M3, M8) | 48 completions per cell (3 prompts × 16 samples), Wilson CIs. Both rows stay post-hoc. |
| manifold-steering (M8) | Bootstrap CIs over source prompts for E_BC and via-intermediate. Rows 1–2 stay post-hoc. |
| jlens, neuron-basis, world-model, emotions (M8) | CIs on the reported numbers: Wilson for swap rates; bootstrap over documents (neuron-basis loss), position bins (world-model r) and activity pairs (emotion preference). |
| belief-fractal (m9) | Deviations documented: d_head 64 (paper 8); Adam, 30k steps × batch 512 (paper SGD, lr 0.01). No rerun. |

## Later batches (criteria are added here before each batch runs)

| Batch | Entries | Models | Where |
|---|---|---|---|
| 2 — circuits | copy-suppression, greater-than, othello-gpt, curve-detectors (+BSF block) | GPT-2 small, TL othello-gpt, Lucent InceptionV1 | local |
| 3 — geometry | calendar-circles, number-helix, linebreak-manifold, belief-fractal, colour-bicone, toy-superposition | GPT-2 small, Pythia 160M–2.8B, Qwen3-1.7B, tiny models, FLUX VAE | local |
| 4 — Neural Geometry series | world-inside (mountain car), manifold-steering, geometric-calculator (probes), sae-manifolds (synthetic), stories-in-space, block-sparse-featurizers | own tiny models, Qwen3-1.7B, Qwen2.5-1.5B-Instruct, Pythia | local |
| 5 — features / steering / lenses | sae-features, actadd, refusal-direction, emotion-concepts, jlens, neuron-basis | GPT-2 small/XL, Qwen2.5-1.5B-Instruct, Qwen3-0.6B/1.7B | local |
| 6 — attribution graphs | attribution-graph (Qwen3-1.7B circuit-tracer; Gemma-2-2B via Neuronpedia API) | Qwen3-1.7B, Gemma-2-2B | local + Neuronpedia |
| 7 — paper scale | calculator + manifold steering + Engels interventions (Llama-3.1-8B or Qwen3-8B/Mistral-7B), helix (Pythia-6.9B/GPT-J), emotions/J-lens/linebreak at 8B (Qwen3-8B), MoE neurons (OLMoE-1B-7B) | 6–8B models | RunPod, ≤ $20 total |
