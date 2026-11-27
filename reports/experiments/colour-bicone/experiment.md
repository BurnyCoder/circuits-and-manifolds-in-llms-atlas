# Experiment: The colour bicone in an image autoencoder

Code [`src/atlas/entries/colour_bicone.py`](../../../src/atlas/entries/colour_bicone.py) (the docstring gives the full method) · site
data `site/public/data/colour-bicone.json` · page `#/entry/colour-bicone` · pre-stated criteria and dated amendments:
[`reports/planned-runs.md`](../../planned-runs.md).

## Question

Do solid colours occupy a 3-D, HSL-like subspace of the FLUX.1 VAE latent (Pach et al. 2026, [arXiv 2603.12261](https://arxiv.org/abs/2603.12261))?

## Setup

400 solid-colour images sampled uniformly in HSV, encoded with the FLUX.1 VAE (`diffusers/FLUX.1-vae`). Latents averaged over space, centred, PCA. Correlations of PC1 with lightness, of the PC2–PC3 angle with hue (circular) and of the radius with chroma. Control: the same pipeline on random-noise images.

## Results vs published claims

<!-- table:replication -->
| Claim | Published | Ours | Pre-stated criterion | Verdict |
|---|---|---|---|---|
| Solid colors occupy a 3-D subspace of the FLUX VAE latent | first three PCs explain essentially 100% of variance (Pach et al. 2026, §3) | top-3 PCs explain 96.14% over 400 colors (random-noise images: 60.1%) | ≥ 95% | ✅ pass |
| The subspace is HSL-like: lightness axis, hue angle, saturation radius | PC1 = lightness; PC2–PC3 angle = hue; radius = saturation (bicone, Fig. 2) | corr(PC1, lightness) 0.98; circular corr(angle, hue) 0.99; corr(radius, chroma) 0.93 | all three ≥ 0.8 | ✅ pass |
<!-- /table:replication -->

## Interpretation

Both claims replicate. Three components carry almost all variance for solid colours, far more than for noise images, and the three axes line up with lightness, hue and saturation. That makes the HSL bicone of the paper.

## Limitations

- Solid colours are the easiest case; textures and lighting are not tested.
- The weights are an ungated copy of the FLUX.1 VAE (same configuration), not the gated official repository.

## All summary values

<!-- table:summary -->
| Key | Value |
|---|---|
| `n_colors` | 400 |
| `var3` | 0.961425 |
| `var1` | 0.503714 |
| `var_noise3` | 0.601373 |
| `r_light` | 0.98306 |
| `r_hue` | 0.989429 |
| `r_sat` | 0.928124 |
| `latent_channels` | 16 |
<!-- /table:summary -->
