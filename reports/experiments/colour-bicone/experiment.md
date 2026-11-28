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
| Claim | Published | Ours | Criterion | Verdict |
|---|---|---|---|---|
| Solid colors occupy a 3-D subspace of the FLUX VAE latent | first three PCs explain essentially 100% of variance (Pach et al. 2026, §3) | top-3 PCs explain 96.14% over 400 colors (random-noise images: 60.1%) | ≥ 95% | ✅ pass |
| The subspace is HSL-like: lightness axis, hue angle, saturation radius | PC1 = lightness; PC2–PC3 angle = hue; radius = saturation (bicone, Fig. 2) | corr(PC1, lightness) 0.98; circular corr(angle, hue) 0.99; corr(radius, chroma) 0.93 | all three ≥ 0.8 | ✅ pass |
| Confirmatory control: the HSL bicone is learned (random-init VAE) | the bicone reflects the trained VAE's colour representation | min(r_light, r_hue, r_sat): trained 0.93 vs random-init VAE 0.08 (0.51 / 0.75 / 0.08); PCA of the RGB values themselves 0.40 (reported) | trained ≥ random-init + 0.2 (pre-stated 2026-11-28) | ✅ pass |
<!-- /table:replication -->

## Interpretation

Both claims pass. Claim 1 is weakly discriminating: a family of inputs with three parameters (hue, saturation,
lightness) is expected to occupy few dimensions in any smooth encoder, and noise images are an easy contrast. Claim 2
is the real test: the three axes line up with lightness, hue and saturation, which makes the HSL bicone of the paper.

**Confirmatory control (pre-stated 2026-11-28): pass.** A randomly initialised VAE with the same architecture
(diffusers `AutoencoderKL.from_config`, weights checked non-constant) scores far lower on the weakest bicone
correlation (`rand_min_r` vs `min_r`), so the structure is learned. The reported linear-RGB baseline (PCA of the colour
values themselves) aligns lightness and chroma almost perfectly but not hue (`rgb_r_light`, `rgb_r_hue`, `rgb_r_sat`).
Its hue score depends on the sample: in the 120-colour smoke run the RGB baseline scored high on all three, and with
400 colours PC1 tilts off the grey axis and the hue angle breaks down. The trained VAE's hue circle is stable across
both sample sizes.

## Limitations

- Solid colours are the easiest case; textures and lighting are not tested.
- The weights are an ungated copy of the FLUX.1 VAE (same configuration), not the gated official repository.
- One random initialisation of the control VAE.

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
| `min_r` | 0.928124 |
| `rand_min_r` | 0.084694 |
| `rand_r_light` | 0.507343 |
| `rand_r_hue` | 0.750573 |
| `rand_r_sat` | 0.084694 |
| `rgb_min_r` | 0.402643 |
| `rgb_r_light` | 0.990867 |
| `rgb_r_hue` | 0.402643 |
| `rgb_r_sat` | 0.995809 |
<!-- /table:summary -->
