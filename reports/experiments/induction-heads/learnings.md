# Learnings: induction heads

- **Single-position ICL scores are too noisy at small n.** Olsson et al.'s ICL score (loss at token 500 − loss at token
  50) over 8 contexts gave −0.05 nats for the final Pythia-160M checkpoint but −1.13 at step 2,000, which is
  implausible. Per-token loss has a standard deviation of several nats, so the standard error over 8 contexts is about
  1 nat. Fix: 128 contexts and 11-token windows around tokens 50 and 500 (documented deviation).
- **Memory.** Full logits for 128 × 512 tokens × 50k vocab are ~13 GB in fp32, so the first full run ran out of memory
  (12.26 GiB request). Fix: compute cross-entropy in chunks of 8 sequences.
- **Choice of ablation.** Zero-ablation is a reasonable choice in an attention-only model, and the random-head control
  uses the same ablation, so the comparison stays fair even if zero is off-distribution.
- **Literature.** TransformerLens's `attn-only-2l` is the model from Neel Nanda's induction-head demos, and ARENA's
  chapter 1.2 defines the same prefix-matching score, so our numbers are directly comparable to those tutorials.
