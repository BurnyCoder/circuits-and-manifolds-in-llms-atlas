# Experiment: SAE features and Golden Gate steering (GPT-2)

Code [`src/atlas/entries/sae_features.py`](../../../src/atlas/entries/sae_features.py) (the docstring gives the full method) · site
data `site/public/data/sae-features.json` · page `#/entry/sae-features` · pre-stated criteria and dated amendments:
[`reports/planned-runs.md`](../../planned-runs.md).

## Question

Can features of an open SAE steer generation toward their concept, as in Golden Gate Claude (Templeton et al. 2024, [Transformer Circuits](https://transformer-circuits.pub/2024/scaling-monosemanticity/))?

## Setup

GPT-2 small, `gpt2-small-res-jb` layer 8. Features chosen by Neuronpedia's explanation search over auto-interp labels (cached; activation-based fallback). Top WikiText activations and logit effects; steering by adding α·W_dec[f] at all positions for α ∈ {0…80}, keyword-based topic adherence, norm-matched random direction as control.

## Results vs published claims

<!-- table:replication -->
| Claim | Published | Ours | Criterion | Verdict |
|---|---|---|---|---|
| Adding an SAE feature's direction steers generation toward its concept (a bridge feature; Golden Gate analogue) | clamping the Golden Gate Bridge feature makes Claude 3 Sonnet obsessed with the bridge (Templeton et al. 2024) | GPT-2 small layer 8, feature #17608: completions mentioning the bridge 0% → 92% at α = 40 (random direction 0%); 'Golden Gate' or 'San Francisco' named in 0% | ≥ 50% on-topic and ≥ 30 points above baseline and random | ✅ pass (post-hoc: feature-selection rule changed twice after smoke tests that showed steering outcomes) |
| Feature steering generalizes across concepts | many features steer behaviour (feature steering section) | 3/5 concepts gain ≥ 30 points of on-topic completions at α = 40 (median lift 58 points) | ≥ 3 of 5 concepts | ✅ pass (post-hoc: feature-selection rule changed twice after smoke tests that showed steering outcomes) |
<!-- /table:replication -->

## Interpretation

Exploratory; both rows are post-hoc. The feature-selection rule changed twice after smoke tests whose logs showed the
steering rates, and only the final rule (Neuronpedia explanation search) passes claim 2. The "Golden Gate" feature is a
generic bridge feature: its auto-interp label is about bridges in general, and the steered completions mention bridges
(the keyword list counts plain "bridge") but none names the Golden Gate or San Francisco (`gg_specific` in the summary
table). So the steering effect is real but it is bridge steering, not a Golden Gate Claude analogue. Three of five
concepts gain substantially at α = 40; French and Python features are identified correctly by their labels but do not
make GPT-2 small write in those styles. Details: [methods review](../../reviews/methods-review.md), M3.

## Limitations

- Keyword adherence is a coarse metric.
- One layer and five concepts; 12 completions per cell and no CIs. To do: ≥ 50 completions per cell with Wilson CIs, and
  all three selection rules' full-run results side by side.

## All summary values

<!-- table:summary -->
| Key | Value |
|---|---|
| `gg_feature` | 17608 |
| `gg_base` | 0.0 |
| `gg_steer` | 0.916667 |
| `gg_random` | 0.0 |
| `gg_specific` | 0.0 |
| `alpha` | 40 |
| `n_concepts` | 5 |
| `n_lift` | 3 |
| `median_lift` | 0.583333 |
<!-- /table:summary -->
