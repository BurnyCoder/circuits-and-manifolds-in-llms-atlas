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
| Claim | Published | Ours | Criterion | Verdict |
|---|---|---|---|---|
| The world model's latents encode the car's state | linear probes: position R² ≈ 0.95, velocity ≈ 0.90 (Wurgaft et al. App. B) | position R² 1.00, velocity 0.06 (untrained encoder: position 0.97) | position R² ≥ 0.8 and ≥ 0.2 above the untrained encoder | ❌ fail |
| Distances along the manifold match distances in the world; straight lines don't | arc-length r = 0.996 (manifold) vs 0.06 (chord) | r = 0.997 along the spline vs 0.369 for straight chords (50 anchors, 1,225 pairs) | manifold r ≥ 0.9 and > chord r | ✅ pass |
| Control: the isometry is specific to the trained model | the manifold reflects learned world structure | untrained encoder: arc r = 0.998, chord r = 0.419 (trained: 0.997 / 0.369) | trained arc-minus-chord gap ≥ 0.2 larger than the untrained encoder's | ❌ fail (post-hoc: control added after the methods review) |
<!-- /table:replication -->

## Interpretation

Neither claim shows learned structure. The isometry row passes its pre-stated criterion, but the control added after
the methods review fails: the untrained encoder's latents give the same arc-length correlation and a similar chord
correlation (`r_arc_untrained`, `r_chord_untrained` vs `r_arc`, `r_chord` in the summary table). The spline is fit
through means binned by the true position, so arc length is close to monotone in position whatever the encoder has
learned; the pass has no discriminating power. Our metric also differs from the paper's, which compares arc lengths on
the activation manifold with arc lengths on the behaviour manifold. The probe claim fails against its pre-stated control
because even an untrained encoder decodes position well: the car is a visible blob. Post hoc, the GRU memory encodes
velocity far better after training (`r2_velocity_gru` vs `r2_velocity_gru_untrained`), which a single frame cannot show;
that is the only training effect seen here. Details: [methods review](../../reviews/methods-review.md), M5.

## Limitations

- Smaller resolution (64×64) than the paper.
- Position decodability and the spline isometry are not evidence of learning when an untrained encoder shows them too.
- To do: the paper's activation-vs-behaviour isometry, and a pixel-PCA baseline.

## All summary values

<!-- table:summary -->
| Key | Value |
|---|---|
| `r2_position` | 0.999259 |
| `r2_velocity` | 0.061259 |
| `r2_position_untrained` | 0.968773 |
| `r2_velocity_gru` | 0.830147 |
| `r2_velocity_gru_untrained` | 0.544898 |
| `r_arc` | 0.996517 |
| `r_chord` | 0.369301 |
| `r_arc_untrained` | 0.997961 |
| `r_chord_untrained` | 0.418801 |
| `void_manifold` | 0.825371 |
| `void_linear` | 1.571868 |
| `n_frames` | 18151 |
| `var3` | 0.56561 |
<!-- /table:summary -->
