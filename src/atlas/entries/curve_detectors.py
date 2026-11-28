"""Curve detectors in InceptionV1 (mixed3b): orientation-tuned neurons that together tile all orientations.

Recomputes the synthetic-stimulus evidence of Cammarata et al. 2020, "Curve Detectors" (Distill,
https://distill.pub/2020/circuits/curve-detectors/): the ten mixed3b curve neurons (379, 406, 385, 343, 342, 388, 324,
340, 330, 349) respond to curves of one orientation, prefer curves to straight lines, tolerate many radii, and their
preferred orientations cover the full circle ("radial tuning curves" / "joint tuning curves" sections).
Model: the original Lucid InceptionV1 weights via Lucent (https://github.com/greentfrapp/lucent, Apache-2.0); Lucid's
preprocessing x*255-117 (lucent.optvis.transform.preprocess_inceptionv1). We draw arcs of radius r whose midpoint sits
at the image center and whose bulge points in direction θ, read each neuron at the center of the 28×28 mixed3b map,
and compare with 20 random non-curve mixed3b neurons (control). Feature visualizations are rendered with Lucent.
"""

from __future__ import annotations

import math
import time

import numpy as np
import torch
from PIL import Image, ImageDraw

from ..common.env import SITE_DATA
from ..common.io import verdict_row, write_entry
from ..common.log import get_logger

log = get_logger()
SLUG = "curve-detectors"
CURVE_NEURONS = [379, 406, 385, 343, 342, 388, 324, 340, 330, 349]
SIZE = 224
ORIENTS = np.arange(0, 360, 5)
RADII = [15, 30, 60, 110]


def arc_image(theta_deg: float, radius: float | None, width: int = 4) -> np.ndarray:
    """Black arc on white, midpoint at the center; bulge toward θ (0° = right, CCW). radius None -> straight line."""
    img = Image.new("L", (SIZE, SIZE), 255)
    d = ImageDraw.Draw(img)
    cx = cy = SIZE / 2
    t = math.radians(theta_deg)
    if radius is None:
        # Straight segment through the center, perpendicular to the bulge direction (the r -> ∞ limit of the arc).
        L = 60
        dx, dy = -math.sin(t) * L, -math.cos(t) * L
        d.line([(cx - dx, cy - dy), (cx + dx, cy + dy)], fill=0, width=width)
    else:
        # Circle center lies opposite the bulge: the arc's outward normal at its midpoint points along θ.
        ox, oy = cx - radius * math.cos(t), cy + radius * math.sin(t)
        span = min(70.0, math.degrees(60 / radius))  # ~120px of arc length, at most ±70°
        start = -theta_deg - span  # PIL angles are clockwise from +x with y down
        d.arc([ox - radius, oy - radius, ox + radius, oy + radius], start, -theta_deg + span, fill=0, width=width)
    a = np.asarray(img, dtype=np.float32) / 255.0
    return np.repeat(a[None], 3, 0)


@torch.no_grad()
def responses(model, images: np.ndarray, device: str, neurons: list[int]) -> np.ndarray:
    """mixed3b activations at the spatial center (max over the central 3x3) for each image: [n_images, n_neurons]."""
    store = {}
    h = model.mixed3b.register_forward_hook(lambda m, i, o: store.__setitem__("a", o))
    out = []
    for i in range(0, len(images), 64):
        x = torch.tensor(images[i : i + 64], device=device) * 255 - 117  # Lucid InceptionV1 preprocessing
        model(x)
        a = store["a"]  # [b, 480, 28, 28]
        c = a.shape[-1] // 2
        out.append(a[:, neurons, c - 1 : c + 2, c - 1 : c + 2].amax((-1, -2)).cpu().numpy())
    h.remove()
    return np.concatenate(out)


def selectivity(r: np.ndarray, theta_deg: np.ndarray) -> float:
    """1 - circular variance of rectified responses over orientation (1 = responds to one orientation only)."""
    r = np.clip(r, 0, None)
    if r.sum() == 0:
        return 0.0
    z = (r * np.exp(1j * np.radians(theta_deg))).sum() / r.sum()
    return float(abs(z))


def feature_vis(model, neurons: list[int], device: str, steps: int) -> list[str]:
    """Render Lucent feature visualizations, save PNGs into site/public/img/curve-detectors/, return relative paths."""
    from lucent.optvis import render

    out_dir = SITE_DATA.parent / "img" / "curve-detectors"
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for n in neurons:
        imgs = render.render_vis(model, f"mixed3b:{n}", thresholds=(steps,), show_image=False, progress=False)
        arr = (np.clip(imgs[-1][0], 0, 1) * 255).astype(np.uint8)
        Image.fromarray(arr).save(out_dir / f"mixed3b_{n}.png")
        paths.append(f"img/curve-detectors/mixed3b_{n}.png")
    return paths


def compute(ctx) -> None:
    from lucent.modelzoo import inceptionv1

    t0 = time.time()
    model = inceptionv1(pretrained=True).to(ctx.device).eval()
    rng = np.random.default_rng(ctx.seed)
    controls = sorted(int(x) for x in rng.choice([n for n in range(480) if n not in CURVE_NEURONS], 20, replace=False))
    neurons = CURVE_NEURONS + controls
    imgs, meta = [], []
    for r in RADII + [None]:
        for th in ORIENTS:
            imgs.append(arc_image(float(th), r))
            meta.append((r, th))
    act = responses(model, np.stack(imgs), ctx.device, neurons)  # [n_img, n_neurons]
    n_or = len(ORIENTS)
    curve = act[: len(RADII) * n_or].reshape(len(RADII), n_or, -1)  # [radius, orient, neuron]
    line = act[len(RADII) * n_or :]  # [orient, neuron]

    per = []
    for j, n in enumerate(neurons):
        best_r = int(curve[:, :, j].max(1).argmax())
        tun = curve[best_r, :, j]
        per.append(
            {
                "neuron": n,
                "is_curve": n in CURVE_NEURONS,
                "best_radius": RADII[best_r],
                "pref_orient": float(ORIENTS[int(tun.argmax())]),
                "selectivity": selectivity(tun, ORIENTS),
                "curve_over_line": float(tun.max() / max(line[:, j].max(), 1e-6)),
                "tuning": curve[:, :, j],  # [radius, orient]
                "line": line[:, j],
            }
        )
    cur = [p for p in per if p["is_curve"]]
    ctl = [p for p in per if not p["is_curve"]]
    prefs = np.sort([p["pref_orient"] for p in cur])
    gaps = np.diff(np.r_[prefs, prefs[0] + 360])
    max_gap = float(gaps.max())
    sel_c = float(np.median([p["selectivity"] for p in cur]))
    sel_n = float(np.median([p["selectivity"] for p in ctl]))
    # Review m2: silent control neurons get selectivity 0 by definition; also report responsive controls only, and a
    # two-sample bootstrap CI of the curve − control median difference (neurons resampled within each group).
    sc, sn = np.array([p["selectivity"] for p in cur]), np.array([p["selectivity"] for p in ctl])
    sn_resp = np.array([p["selectivity"] for p in ctl if p["tuning"].max() > 0])
    brng = np.random.default_rng(ctx.seed)
    boots = [np.median(brng.choice(sc, len(sc))) - np.median(brng.choice(sn, len(sn))) for _ in range(2000)]
    diff_lo, diff_hi = (float(x) for x in np.quantile(boots, [0.025, 0.975]))
    n_curve_gt_line = sum(p["curve_over_line"] > 1.5 for p in cur)
    fv = feature_vis(model, CURVE_NEURONS, ctx.device, 64 if ctx.quick else 512)
    log.info("curve detectors: selectivity curve %.2f vs control %.2f; max gap %.0f°; curve>1.5×line %d/10", sel_c, sel_n, max_gap, n_curve_gt_line)

    summary = {
        "sel_curve_median": sel_c,
        "sel_control_median": sel_n,
        "max_orient_gap": max_gap,
        "n_curve_gt_line": n_curve_gt_line,
        "n_controls": len(ctl),
        "sel_control_resp_median": float(np.median(sn_resp)) if len(sn_resp) else float("nan"),
        "n_controls_resp": int(len(sn_resp)),
        "sel_diff_lo": diff_lo,
        "sel_diff_hi": diff_hi,
    }
    repl = [
        verdict_row(
            "Curve neurons are tuned to one orientation",
            "each curve detector responds to curves in a narrow orientation band (radial tuning curves)",
            f"median orientation selectivity {sel_c:.2f} vs {sel_n:.2f} for 20 random mixed3b neurons (difference 95% CI [{diff_lo:.2f}, {diff_hi:.2f}]); "
            f"{summary['sel_control_resp_median']:.2f} over the {len(sn_resp)} responsive controls",
            "curve median ≥ control median + 0.2",
            sel_c >= sel_n + 0.2,
        ),
        verdict_row(
            "Curve neurons prefer curves to straight lines",
            "near-zero activations are mostly straight lines; synthetic curves reach up to 24σ",
            f"{n_curve_gt_line}/10 neurons respond > 1.5× more to their best curve than to any straight line",
            "≥ 8 of 10",
            n_curve_gt_line >= 8,
            partial=n_curve_gt_line >= 5,
        ),
        verdict_row(
            "Together they tile all orientations",
            "the family covers the full circle (joint tuning curves)",
            f"largest gap between preferred orientations {max_gap:.0f}°",
            "≤ 90°",
            max_gap <= 90,
        ),
    ]
    payload = {"radii": RADII, "orients": ORIENTS, "neurons": per, "feature_vis": dict(zip(map(str, CURVE_NEURONS), fv))}
    write_entry(SLUG, payload, summary, repl, models=["InceptionV1 (Lucid weights via Lucent)"], runtime_s=time.time() - t0, ndigits=3)
