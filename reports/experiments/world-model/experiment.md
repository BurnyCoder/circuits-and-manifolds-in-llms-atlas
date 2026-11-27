# Experiment: The world inside a world model (trained here)

Code [`src/atlas/entries/world_model.py`](../../../src/atlas/entries/world_model.py) (the docstring gives the full method) · site
data `site/public/data/world-model.json` · page `#/entry/world-model` · pre-stated criteria and dated amendments:
[`reports/planned-runs.md`](../../planned-runs.md).

## Question

Does a mountain-car world model embed frames on a smooth curve indexed by the car's position, so that distances along the manifold match distances in the world (Geiger et al. 2026; Wurgaft et al. 2026, App. B)?

## Setup

MountainCar-v0, 100 rollouts with mixed policies; conv encoder → 64-D LayerNorm latent, action embedding, GRU-64, conv decoders, trained 4,000 steps at 64×64. Probes for position and velocity (rollout-level split) vs an untrained encoder; a smoothing spline through position-binned latent means; isometry of arc length vs world distance over 50 anchors, compared with straight chords.

## Results vs published claims

<!-- table:replication -->
| Claim | Published | Ours | Pre-stated criterion | Verdict |
|---|---|---|---|---|
| The world model's latents encode the car's state | linear probes: position R² ≈ 0.95, velocity ≈ 0.90 (Wurgaft et al. App. B) | position R² 1.00, velocity 0.06 (untrained encoder: position 0.97) | position R² ≥ 0.8 and ≥ 0.2 above the untrained encoder | ❌ fail |
| Distances along the manifold match distances in the world; straight lines don't | arc-length r = 0.996 (manifold) vs 0.06 (chord) | r = 0.995 along the spline vs 0.248 for straight chords (50 anchors, 1,225 pairs) | manifold r ≥ 0.9 and > chord r | ✅ pass |
<!-- /table:replication -->

## Interpretation

The headline result replicates closely. Arc length along the latent manifold tracks world distance almost perfectly, while straight chords do not. The probe claim fails against its pre-stated control because even an untrained encoder decodes position well: the car is a visible blob. Post hoc, the GRU memory encodes velocity far better after training, which a single frame cannot show.

## Limitations

- Smaller resolution (64×64) than the paper.
- Position decodability is not evidence of learning when an untrained encoder already decodes it.

## All summary values

<!-- table:summary -->
| Key | Value |
|---|---|
| `r2_position` | 0.998551 |
| `r2_velocity` | 0.061323 |
| `r2_position_untrained` | 0.968773 |
| `r2_velocity_gru` | 0.874276 |
| `r2_velocity_gru_untrained` | 0.544898 |
| `r_arc` | 0.994983 |
| `r_chord` | 0.247711 |
| `void_manifold` | 0.775999 |
| `void_linear` | 1.327291 |
| `n_frames` | 18151 |
| `var3` | 0.62875 |
<!-- /table:summary -->
