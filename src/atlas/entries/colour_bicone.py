"""Color lives on a hue–saturation–lightness bicone inside the FLUX image autoencoder's latent space.

Recomputes the discovery step of Pach, Bader, Bouniot, Belongie, Akata 2026, "The Latent Color Subspace: Emergent
Order in High-Dimensional Chaos" (https://arxiv.org/abs/2603.12261, ICML 2026, CC BY 4.0; code
https://github.com/ExplainableML/LCS): encode solid-color images with the FLUX VAE, average each image's latent over
space, center and run PCA — the first three components explain essentially all variance; PC1 tracks lightness, the
angle in the PC2–PC3 plane tracks hue and the radius tracks saturation (an HSL-like bicone, paper Fig. 2).
Weights: ``diffusers/FLUX.1-vae`` (the FLUX.1 autoencoder in diffusers format, published ungated by the Hugging Face
diffusers team; config latent_channels 16, scaling 0.3611, shift 0.1159 = the FLUX.1 VAE). The paper used FLUX.1-dev's
VAE; the official black-forest-labs repos are gated behind a click-through, so we use this copy and note it as a caveat.
Only config.json + safetensors are loaded (the repo's endpoint handler.py is never executed).
Control: the same pipeline on random-noise images.
"""

from __future__ import annotations

import colorsys
import time

import numpy as np
import torch

from ..common.geometry import circular_corr, pca
from ..common.io import verdict_row, write_entry
from ..common.log import get_logger

log = get_logger()
SLUG = "colour-bicone"
REPO = "diffusers/FLUX.1-vae"


@torch.no_grad()
def encode_mean(vae, imgs: np.ndarray, device: str, bs: int = 16) -> np.ndarray:
    """Spatially averaged latent mean for each image ([n, 3, H, W] in [0, 1]) -> [n, latent_channels]."""
    out = []
    for i in range(0, len(imgs), bs):
        x = torch.tensor(imgs[i : i + bs], device=device, dtype=vae.dtype) * 2 - 1  # VAE expects [-1, 1]
        lat = vae.encode(x).latent_dist.mean  # [b, C, h, w]
        out.append(lat.float().mean((2, 3)).cpu().numpy())
    return np.concatenate(out)


def compute(ctx) -> None:
    from diffusers import AutoencoderKL

    t0 = time.time()
    vae = AutoencoderKL.from_pretrained(REPO, torch_dtype=torch.float16 if ctx.device == "cuda" else torch.float32).to(ctx.device).eval()
    rng = np.random.default_rng(ctx.seed)
    n = 120 if ctx.quick else 400
    hsv = np.c_[rng.random(n), rng.random(n), rng.random(n)]  # uniform in HSV, as in the paper
    rgb = np.array([colorsys.hsv_to_rgb(*c) for c in hsv])
    hls = np.array([colorsys.rgb_to_hls(*c) for c in rgb])  # (hue, lightness, saturation)
    S = 128
    imgs = np.broadcast_to(rgb[:, :, None, None], (n, 3, S, S)).astype(np.float32).copy()
    Z = encode_mean(vae, imgs, ctx.device)
    scores, comps, var, mu = pca(Z, 3)
    # Orient PC1 so that it increases with lightness.
    if np.corrcoef(scores[:, 0], hls[:, 1])[0, 1] < 0:
        scores[:, 0] *= -1
    r_light = float(np.corrcoef(scores[:, 0], hls[:, 1])[0, 1])
    chroma = (hls[:, 2] > 0.3) & (hls[:, 1] > 0.2) & (hls[:, 1] < 0.8)  # clearly colored samples
    ang = np.arctan2(scores[:, 2], scores[:, 1])
    r_hue = abs(circular_corr(ang[chroma], 2 * np.pi * hls[chroma, 0]))
    radius = np.hypot(scores[:, 1], scores[:, 2])
    chroma_true = hls[:, 2] * (1 - np.abs(2 * hls[:, 1] - 1))  # HSL chroma: saturation scaled by the bicone width
    r_sat = float(np.corrcoef(radius, chroma_true)[0, 1])
    # Control: random-noise images through the same pipeline.
    noise = rng.random((n, 3, S, S)).astype(np.float32)
    _, _, var_noise, _ = pca(encode_mean(vae, noise, ctx.device), 3)
    log.info("FLUX VAE colour PCA var %s (noise %s); r_light %.3f r_hue %.3f r_sat %.3f", np.round(var, 4), np.round(var_noise, 4), r_light, r_hue, r_sat)
    summary = {
        "n_colors": n,
        "var3": float(var.sum()),
        "var1": float(var[0]),
        "var_noise3": float(var_noise.sum()),
        "r_light": r_light,
        "r_hue": r_hue,
        "r_sat": r_sat,
        "latent_channels": int(Z.shape[1]),
    }
    repl = [
        verdict_row(
            "Solid colors occupy a 3-D subspace of the FLUX VAE latent",
            "first three PCs explain essentially 100% of variance (Pach et al. 2026, §3)",
            f"top-3 PCs explain {var.sum() * 100:.2f}% over {n} colors (random-noise images: {var_noise.sum() * 100:.1f}%)",
            "≥ 95%",
            var.sum() >= 0.95,
        ),
        verdict_row(
            "The subspace is HSL-like: lightness axis, hue angle, saturation radius",
            "PC1 = lightness; PC2–PC3 angle = hue; radius = saturation (bicone, Fig. 2)",
            f"corr(PC1, lightness) {r_light:.2f}; circular corr(angle, hue) {r_hue:.2f}; corr(radius, chroma) {r_sat:.2f}",
            "all three ≥ 0.8",
            min(r_light, r_hue, r_sat) >= 0.8,
            partial=min(r_light, r_hue) >= 0.8,
        ),
    ]
    payload = {"pcs": scores, "rgb": (rgb * 255).round().astype(int), "hls": hls, "var": var, "var_noise": var_noise}
    write_entry(SLUG, payload, summary, repl, models=[f"{REPO} (VAE)"], runtime_s=time.time() - t0, ndigits=3)
    del vae
    torch.cuda.empty_cache()
