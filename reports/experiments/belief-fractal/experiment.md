# Experiment: Belief-state fractal (trained here)

Code [`src/atlas/entries/belief_fractal.py`](../../../src/atlas/entries/belief_fractal.py) (the docstring gives the full method) · site
data `site/public/data/belief-fractal.json` · page `#/entry/belief-fractal` · pre-stated criteria and dated amendments:
[`reports/planned-runs.md`](../../planned-runs.md).

## Question

Does a small transformer trained on the Mess3 hidden Markov process linearly represent the Bayes-optimal belief state (a fractal on the simplex) in its final residual stream (Shai et al. 2024, [arXiv 2405.15943](https://arxiv.org/abs/2405.15943))?

## Setup

4-layer transformer (d = 64, 1 head, LayerNorm) trained for 30,000 steps on Mess3 (x = 0.05, α = 0.85) sequences of 10 tokens. Linear least-squares readout of beliefs from the final residual stream on held-out sequences across checkpoints, compared with the identical initial weights (random-init control); model loss vs the exact Bayes-optimal loss on the same held-out sequences.

## Results vs published claims

<!-- table:replication -->
| Claim | Published | Ours | Pre-stated criterion | Verdict |
|---|---|---|---|---|
| Belief states are linearly decodable from the final residual stream | Mess3: linear regression recovers the belief fractal (MSE ≈ 4e-4, paper Fig. 3) | held-out R² 0.975 (MSE 3.99e-03) after 30000 steps vs 0.932 at initialization | R² ≥ 0.9 and ≥ 0.3 above the random-init control | ❌ fail |
| The model approaches the optimal predictor | trained to near the process's entropy rate | final loss 0.8034 nats vs optimal 0.8032 and uniform 1.0986 | excess loss < 10% of the uniform-to-optimal gap | ✅ pass |
<!-- /table:replication -->

## Interpretation

The model reaches the Bayes-optimal loss almost exactly, and its residual stream holds the belief geometry (high held-out R², fractal readout on the site). The pre-stated criterion still fails because the random-init control is already strong. In Mess3 each token is highly informative about the hidden state, so the last token's embedding, present in any residual stream, linearly predicts most of the belief. Post hoc, training cuts the readout MSE several-fold relative to initialization, which is the more discriminating comparison.

## Limitations

- The pre-stated control turned out too easy for this process; a harder process (e.g. RRXOR) or an MSE-ratio criterion would discriminate better.
- One seed.

## All summary values

<!-- table:summary -->
| Key | Value |
|---|---|
| `steps` | 30000 |
| `final_r2` | 0.975239 |
| `final_mse` | 0.003993 |
| `rand_r2` | 0.932394 |
| `rand_mse` | 0.010903 |
| `all_layers_r2` | 0.987856 |
| `final_loss` | 0.803387 |
| `optimal_loss` | 0.803162 |
| `uniform_loss` | 1.098612 |
<!-- /table:summary -->
