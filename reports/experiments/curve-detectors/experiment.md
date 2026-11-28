# Experiment: curve detectors (InceptionV1 mixed3b)

Code [`src/atlas/entries/curve_detectors.py`](../../../src/atlas/entries/curve_detectors.py) · site data
`site/public/data/curve-detectors.json` + feature visualizations `site/public/img/curve-detectors/` · page
`#/entry/curve-detectors` · fidelity 🟢 recomputed.

## Question

Do the ten mixed3b neurons that Cammarata et al. 2020 ([Distill, Curve Detectors](https://distill.pub/2020/circuits/curve-detectors/))
call curve detectors (379, 406, 385, 343, 342, 388, 324, 340, 330, 349) respond selectively to curves of one
orientation, prefer curves over straight lines, and together cover all orientations?

## Setup

- **Model:** original Lucid InceptionV1 weights via Lucent (Apache-2.0), preprocessing x·255 − 117.
- **Stimuli:** 224×224 black 4-px arcs on white (about 120 px of arc, at most ±70°), radius r ∈ {15, 30, 60, 110} px,
  bulge direction θ every 5° (72 orientations), arc midpoint at the image centre; 120-px straight lines through the
  centre at the same orientations (the r → ∞ limit).
- **Response:** activation at the centre of the 28×28 mixed3b map.
- **Orientation selectivity:** 1 − circular variance of the rectified tuning curve over θ at each neuron's best radius
  (1 = responds to one orientation only).
- **Tiling:** largest angular gap between the ten neurons' preferred orientations.
- **Feature visualization:** Lucent `render_vis`, 512 optimisation steps per neuron.

**Controls.** 20 random non-curve mixed3b neurons (seeded) for selectivity; straight lines for curve preference.

**Why these settings.** The radii span the receptive-field sizes in the paper's radial tuning figure; 5° steps resolve
the ~20–30° tuning widths reported there.

## Results vs published claims

<!-- table:replication -->
| Claim | Published | Ours | Criterion | Verdict |
|---|---|---|---|---|
| Curve neurons are tuned to one orientation | each curve detector responds to curves in a narrow orientation band (radial tuning curves) | median orientation selectivity 0.95 vs 0.10 for 20 random mixed3b neurons (difference 95% CI [0.75, 0.90]); 0.11 over the 18 responsive controls | curve median ≥ control median + 0.2 | ✅ pass |
| Curve neurons prefer curves to straight lines | near-zero activations are mostly straight lines; synthetic curves reach up to 24σ | 10/10 neurons respond > 1.5× more to their best curve than to any straight line | ≥ 8 of 10 | ✅ pass |
| Together they tile all orientations | the family covers the full circle (joint tuning curves) | largest gap between preferred orientations 65° | ≤ 90° | ✅ pass |
<!-- /table:replication -->

## Interpretation

The ten neurons are sharply orientation-tuned compared with random mixed3b neurons, every one prefers its best curve to
any straight line, and their preferred orientations leave no gap larger than the criterion — they tile the circle, as
in the paper's radial tuning curves. The review (m2) noted that random neurons that never respond get selectivity 0 by
definition, which would flatter the comparison. Restricted to responsive control neurons (`n_controls_resp` of 20), the
control median barely moves (`sel_control_resp_median`), and the curve − control difference has a 95% bootstrap CI of
`sel_diff_lo`–`sel_diff_hi` (neurons resampled within each group), far above the 0.2 criterion. This replicates the synthetic-stimulus part of the Distill article; the dataset-example
and circuit (weights from mixed3a) analyses are not recomputed here.

## Limitations

- Synthetic stimuli only; responses measured at one spatial position.

## All summary values

<!-- table:summary -->
| Key | Value |
|---|---|
| `sel_curve_median` | 0.947427 |
| `sel_control_median` | 0.097087 |
| `max_orient_gap` | 65.0 |
| `n_curve_gt_line` | 10 |
| `n_controls` | 20 |
| `sel_control_resp_median` | 0.107621 |
| `n_controls_resp` | 18 |
| `sel_diff_lo` | 0.747559 |
| `sel_diff_hi` | 0.901043 |
<!-- /table:summary -->
