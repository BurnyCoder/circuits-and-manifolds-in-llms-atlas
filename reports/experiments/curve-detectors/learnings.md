# Learnings: curve detectors

- **Preprocessing.** Lucid InceptionV1 expects inputs in [0, 255] minus 117 (`lucent.optvis.transform.
  preprocess_inceptionv1`); we apply it explicitly to our stimuli.
- **Bulge geometry.** PIL arc angles run clockwise with y pointing down, so the circle centre is placed opposite the
  bulge direction (`arc_image`). Preferred orientations were cross-checked against the feature visualizations.
- **Controls make the number meaningful.** 0.95 selectivity on its own says little. Random mixed3b neurons give 0.10.
