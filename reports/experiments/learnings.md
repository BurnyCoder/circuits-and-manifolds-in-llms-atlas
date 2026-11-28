# Learnings (running log)

What we tried, what went wrong, why, and how it was fixed. Newest last. Per-entry details live in
`reports/experiments/<slug>/learnings.md`.

## 2026-11-27 — environment

- **TransformerLens 4.x removed `HookedTransformer`.** circuit-tracer 0.5.5 pins `transformer-lens<4`, SAELens 6.56 needs
  `>=3`; we pinned TL 3.9.0, which also works with transformers 5.24. Source: the libraries' PyPI metadata and the TL
  v4 migration guide (https://transformerlensorg.github.io/TransformerLens/content/migrating_to_v4.html).
- **transformers 5 renamed `torch_dtype` → `dtype`** in `from_pretrained`; fixed in the Pythia loader.
- **C: drive almost full** → all Hugging Face / torch / uv caches on `D:\AI\circuits-atlas\` via `.env`; uv needs
  `UV_LINK_MODE=copy` across drives.

## 2026-11-27 — quick smoke tests (`--quick`), before full runs

- **Induction / ICL score noise.** With 8 contexts, the single-position ICL score (loss at token 500 − token 50,
  Olsson et al.) gave an implausible −0.05 nats for the final Pythia-160M checkpoint vs −1.13 at step 2,000. Per-token
  loss has a standard deviation of a few nats, so a mean over 8 contexts is too noisy. Fix: 128 contexts and 11-token
  windows around tokens 50 and 500. This changes the estimator, not the induction-score criterion used for the verdict.
- **Greater-than faithfulness method bug.** Our first "circuit-only" test patched all non-circuit heads *and MLPs 0–7*
  with activations from the "1701" prompt and got PD −53% (vs +90% clean). Cause: at the YY token, early MLPs are part of
  the token's embedding pathway (cf. the "effective embedding" W_E + MLP0 of McDougall et al. 2023), so patching them
  deletes the start year — the circuit's input — rather than testing the circuit. Fix: circuit-only = patch the 137
  non-circuit heads, leave MLPs clean. The pass criterion (≥ 60% of full PD) is unchanged and was recorded in
  `planned-runs.md` before the full run.
- **Browser quirk.** `useEffect(() => window.scrollTo(0, 0))` crashed React ("destroy is not a function") in the
  desktop app's Chromium, where `scrollTo` returns a Promise; effects must return nothing or a cleanup function.

## 2026-11-27 — batch 3–5 smoke tests and site QA

- **Small Pythia models cannot add.** In the few-shot `a+b=` prompt, Pythia-410m solved 1 of 59 prompts (e.g.
  23+45 → "53"). The NeoX tokenizer was checked and has single tokens for 0–99, so this is a capability limit, not a
  bug. Patching needs correctly solved prompts, so the patching claim now requires ≥ 20 of them; otherwise it is
  n/a (dated amendment in `planned-runs.md`). Kantamneni & Tegmark used GPT-J-6B, which is why that model is in the
  RunPod tier.
- **Attention-sink outliers distort geometry.** In Qwen3-1.7B the first token carries "massive activations" (Sun et
  al. 2024, arXiv 2402.17762). Including it put 99.9% of the per-count-mean variance in 3 PCs. Position 0 is now
  excluded from the linebreak analysis.
- **Fixed ridge strength overfits wide probes.** α = 10 on 2,048 standardized features gave negative held-out R² in
  late layers. `ridge_r2` now accepts a grid and picks α on held-out *training* groups (documents/stories), never on
  test data (cf. sklearn `RidgeCV`). There is a unit test.
- **Moving HF_HOME hides the login token.** With `HF_HOME` on `D:` the pipeline sent unauthenticated requests. Setting
  `HF_TOKEN_PATH` to the token written by `hf auth login` fixed it without copying the secret. Even authenticated,
  `black-forest-labs/FLUX.1-schnell` is gated by click-through terms on the user's account. We switched to the ungated
  `diffusers/FLUX.1-vae` (same FLUX.1 VAE config) rather than accept terms on the user's behalf.
- **Compare losses on the same data.** The Mess3 model's "final loss" (mean of noisy training batches) came out below
  a separately sampled entropy-rate estimate. Both are now computed on the same held-out sequences (model vs exact
  Bayes predictor).
- **Read the method off the primary source.** For neuron sparsity, Goodfire's layer top-320 keeps neurons by
  absolute activation and *zeroes* the rest (their band experiments mean-ablate). ericzhou12's KL ≈ 0.80 uses a gentler
  rule (distance from mean, mean-fill). Our pre-stated threshold leaned on that number, so the gentler rule is now
  evaluated as a labelled post-hoc comparison.
- **Random-init controls can be strong.** For Mess3, a random-init transformer's residual already predicts beliefs
  with R² ≈ 0.93, because single tokens are informative about the hidden state. The pre-stated "≥ 0.3 above random
  init" criterion will likely fail. It stays as written; the MSE ratio is reported post-hoc.
- **Viz medians.** Three visualizations picked `sorted[floor(n/2)]`, the upper-middle element for even n, and showed
  2.49 where the pipeline's `np.median` gave 2.48. They now use `d3.median`.
- **Browser cache.** Re-generated JSON did not show until a hard reload. Data fetches now use `cache: 'no-cache'`
  (ETag revalidation).
- **Simulation fidelity.** The linebreak "particle" redraw used linear springs, which settle into a smooth arc with no
  ringing. Fitting the paper's ideal similarity (1 within ±w, 0 outside) by gradient descent reproduces ringing, as the
  Gibbs phenomenon of a truncated box kernel predicts. This was checked numerically before changing the site.
- **Pages never deployed.** The first Actions run failed because data files were not committed yet, and the content
  test requires them. Entries awaiting full runs are now flagged `status: pending`. The page shows a notice and the
  test skips only those entries.
- **VRAM spill makes runs 10× slower without an error.** GPT-2 XL in fp32 (6.4 GB) plus activations filled the 8 GB GPU,
  and Windows silently spilled to shared memory: the ActAdd smoke test took 54 min. Loading in bfloat16 and batching
  samples with the KV cache fixes it. The steering hook now fires on the prompt pass only, which is token-identical to
  recomputing the steered positions every step (checked with greedy decoding).
- **Tokenizers decide which probes are possible.** Qwen3 splits numbers into digits, so "21" and "42" are not single
  tokens ("42" and "49" share the first token "4"). The J-lens arithmetic example therefore cannot be scored by token
  rank in this model and stays as a picture only.
- **Simplified interventions can understate an effect.** Exchanging projections on two non-orthogonal unit directions
  is weaker than the paper's coefficient swap in a sparse non-negative J-space decomposition. Both are now reported
  (dated amendment) rather than replacing the pre-stated one.
- **Log what the models say.** Story generation logged only a sample, so every prompt and output is now logged in full.
  The one deliberate exception is refusal-ablated completions of harmful prompts: only the prompt and the refusal flag
  are logged.
- **Picking SAE features by hand-written probes is fragile.** One focus token picks word-sense features: French " pain"
  (bread) gave a violence feature. Requiring activation on three probes picked broad features: a place-name feature
  beat the bridge feature. Neuronpedia's explanation search over auto-interp labels (independent of the steering test)
  finds the intended features: bridge #17608, DNA, Harry Potter, French, Python/Django. It is cached for reproducibility.
- **Windows console encoding.** Log lines with "α" raised `'charmap' codec can't encode` on the console handler, so
  terminal output was lost while the file log was fine. `sys.stdout.reconfigure(encoding="utf-8")` fixes it.
- **Check a verdict cannot pass on degenerate data.** SAE manifolds printed "capture peaks at k = 3: pass" while capture
  was 0% at every k (argmax of zeros). Peak-type claims now use a metric with real variation (restricted R²).
- **Evaluate in distribution.** Showing a manifold alone to a TopK SAE always activates k atoms, so "few atoms per
  manifold" was impossible by construction. The paper evaluates on superposed inputs and attributes rows by ground-truth
  masks. A thresholded support statistic also needs enough instances (48, not 16) so co-occurring manifolds stay under
  the threshold.
- **Re-read the paper before stating a directional claim.** We had pre-stated "larger k shatters". The paper says
  small k shatters and large k dilutes. The amendment records the error.
- **Quick mode hides memory blow-ups.** Caching reference log-probs is 0.5 GB in quick mode but ~7 GB in full mode
  (48 × 512 tokens × 152k vocabulary). The OOM then crashed the runner's cleanup and ended the whole batch. Now the
  reference is recomputed per sequence, parameters are frozen for gradient × activation, the cleanup is exception-safe,
  and each remaining entry runs in its own process.

## 2026-11-28 — adversarial methods review and fact-check

The full audit is in [`../reviews/methods-review.md`](../reviews/methods-review.md) and
[`../reviews/fact-check.md`](../reviews/fact-check.md). What we learned:

- **Iterating an analysis until it passes is the main risk, even with good intentions.** Attribution-graphs went 0/3 →
  1/3 → 3/3 across three full runs while the path search was edited; sae-manifolds and manifold-steering changed a
  metric or a layer after a full run; sae-features changed its selection rule after smoke tests that logged steering
  rates. Each change had a stated reason, but the reasons were found after seeing the outcome. Fix: one auditable list
  of post-hoc rows (`src/atlas/common/post_hoc.py`), shown with the reason and excluded from the tally. Next time:
  commit the analysis before every non-quick run and have the runner refuse to start with a dirty tree.
- **Smoke tests leak outcomes.** A smoke test that prints the verdict metric is a peek. Smoke tests should check
  shapes, timing and degenerate values, not log the number the criterion is about.
- **Check that a control is actually random.** `HookedTransformer(model.cfg)` on a pretrained config builds an
  all-zero network because TransformerLens sets `init_weights=False` for pretrained configs. The saved data showed it
  (the Othello "random-init" probe equalled the shuffled-label probe at every layer, and induction scores sat at the
  uniform-attention value), but we read it as a finding. Properly initialised, the Othello control rises from 54% to
  72%; the verdicts survive.
- **A control must be able to fail.** Projecting out a random unit direction removes about 1/d of the variance;
  shuffling orientations destroys the smoothness that any vision network has; a random plane without norm matching
  moves the activation 1–2 orders of magnitude less than the real patch. Three new controls were added; two of them
  fail and change the reading of a "pass": random InceptionV1 channels are nearly as orientation-harmonic as the curve
  neurons (block-sparse featurizers), and an untrained encoder gives the same spline isometry as the trained world model.
- **Dimension-match baselines.** The helix patch looked good against a 1-D linear patch, but a 9-D PCA patch does as
  well, unlike in the paper.
- **Check keyword metrics against the outputs.** "Golden Gate" steering counted any "bridge"; 0 of 24 steered
  completions mentioned Golden Gate or San Francisco. The specific rate is now reported separately.
- **Read the replication you cite, not its abstract.** Five "replicated by others" rows overstated the source (e.g.
  the Goodfire greater-than post reports clear differences from the original mechanism, and zroe1's phase diagrams only
  roughly match theory). Quotes found in a lab's blog post were attributed to its arXiv paper.
- **Promise only the statistics you compute.** The README promised 95% bootstrap CIs everywhere; three entries had
  them. Wilson intervals (`common.stats.wilson_ci`) are now added to rate-based rows as entries are rerun.
