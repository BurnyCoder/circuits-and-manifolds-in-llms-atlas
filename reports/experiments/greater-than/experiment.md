# Experiment: greater-than (GPT-2 small)

Code [`src/atlas/entries/greater_than.py`](../../../src/atlas/entries/greater_than.py) · site data
`site/public/data/greater-than.json` · page `#/entry/greater-than` · fidelity 🟢 recomputed.

## Question

Does GPT-2 small compute "a year greater than YY" with the circuit of Hanna, Liu, Variengien 2023
([arXiv 2305.00586](https://arxiv.org/abs/2305.00586)) — attention heads 5.1, 5.5, 6.9, 7.10, 8.8, 8.11, 9.1 moving the
start year to the END token and MLPs 8–11 producing the "greater than" distribution?

## Setup

- **Data:** 12 nouns × 97 start years (1702–1798; every one tokenizes as '17' + 'YY' in GPT-2)
  = 1,164 prompts "The {noun} lasted from the year 17YY to the year 17".
- **Metric:** probability difference PD = Σ_{y>YY} p(y) − Σ_{y≤YY} p(y) over the 100 two-digit tokens (paper: 81.7%).
- **Patching:** replace components with activations from the "1701" corrupted prompt (the paper's 01-dataset) for
  every on/off combination of {circuit heads, MLP8, MLP9, MLP10, MLP11}; processed in chunks of 128 prompts.
- **Circuit-only:** patch the 137 non-circuit heads, keep circuit heads and all MLPs clean.

**Controls.** 7 random heads patched (30 draws, median PD).

## Results vs published claims

<!-- table:replication -->
| Claim | Published | Ours | Pre-stated criterion | Verdict |
|---|---|---|---|---|
| GPT-2 small predicts years greater than the start year | probability difference 81.7% | PD = 90.2% [89.7, 90.7] over 1164 prompts | PD > 50% | ✅ pass |
| The identified heads + MLPs 8–11 carry the computation | patching the circuit with '01' activations destroys PD; circuit alone keeps 89.5% (72.7 of 81.7 points) | heads patched → -53.4%, MLPs 8–11 patched → -20.9%, both → -49.7%; circuit-only → 83.4% (92% of full) | patching heads+MLPs drops PD below 30% of full, and circuit-only keeps ≥ 60% of full | ✅ pass |
| Random heads are not the circuit | (control, ours) | patching 7 random heads → median PD 89.6% (circuit heads → -53.4%) | random median PD > circuit-heads PD + 10 points | ✅ pass |
<!-- /table:replication -->

## Interpretation

GPT-2 small does greater-than reliably (PD higher than the paper's, plausibly because our nouns differ). Patching the 7
circuit heads alone reverses PD; patching MLPs 8–11 alone also removes most of it — both stages are needed, as the paper
argues. Seven random heads leave PD essentially unchanged. Keeping only circuit heads clean preserves most of the full
PD.

## Deviations from the paper

- Circuit-only test patches non-circuit heads only (patching MLPs 0–7 deleted the start year itself; see learnings).

## Limitations

- One corrupted baseline ("01"); the heatmap and PD are sensitive to the noun set.

## All summary values

<!-- table:summary -->
| Key | Value |
|---|---|
| `n_prompts` | 1164 |
| `n_years` | 97 |
| `pd_clean` | 0.902217 |
| `pd_corrupt` | -0.516477 |
| `pd_heads_patched` | -0.534092 |
| `pd_mlps_patched` | -0.208946 |
| `pd_all_patched` | -0.497471 |
| `pd_circuit_only` | 0.833723 |
| `faithful_frac` | 0.924082 |
| `rand_pd_median` | 0.896398 |
<!-- /table:summary -->
