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
