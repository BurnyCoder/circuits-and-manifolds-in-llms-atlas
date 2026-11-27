"""Steering along the manifold vs. straight lines (Wurgaft et al. 2026), on Qwen3-1.7B weekday arithmetic.

Paper: Wurgaft, Rager, Kowal, et al., "Manifold Steering Reveals the Shared Geometry of Neural Network Representation and
Behavior" (https://arxiv.org/abs/2605.05115; Goodfire post https://www.goodfire.ai/research/manifold-steering). In
Llama-3.1-8B (layer 28, 64-d PCA) they fit a periodic cubic spline through the activation centroids of the 7 weekday
answers and steer from one day to another either along the spline or along the straight chord. Along the manifold the
output distribution moves through the intermediate days and stays near the behaviour manifold (Bhattacharyya energy
E_BC 0.34 vs 0.93 for linear on weekdays; isometry r .99 vs .89).
Here: prompts "Let's do some day of the week math. {N} days from {D} is" (answers = 7 days), last-token activations at a
middle layer, 64-d PCA, periodic spline (scipy CubicSpline, bc_type='periodic') through the 7 answer centroids. For every
ordered pair of days 2–3 apart we replace the 64-d component of a source prompt's activation with points on (a) the spline
arc and (b) the chord, read the 7-day output distribution, and measure: E_BC = mean Bhattacharyya distance to the
behaviour manifold (spline through the √p centroids), and the fraction of path points whose argmax is an intermediate day.
"""

from __future__ import annotations

import time

import numpy as np
import torch
from scipy.interpolate import CubicSpline

from ..common.io import verdict_row, write_entry
from ..common.log import get_logger
from .calendar_circles import DAYS, NUMWORDS

log = get_logger()
SLUG = "manifold-steering"
MODEL = "Qwen/Qwen3-1.7B"
LAYERS = [12, 16, 20]
K = 13  # points per path


def periodic_spline(points: np.ndarray) -> CubicSpline:
    t = np.arange(len(points) + 1)
    return CubicSpline(t, np.vstack([points, points[:1]]), bc_type="periodic")


def compute(ctx) -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(ctx.device).eval()
    day_ids = [tok(d, add_special_tokens=False).input_ids[0] for d in DAYS]
    prompts = [(k, d, f"Let's do some day of the week math. {NUMWORDS[k - 1]} days from{DAYS[d]} is") for k in range(1, 7) for d in range(7)]
    results = []
    for L in LAYERS if not ctx.quick else LAYERS[:1]:
        H, P, ans = [], [], []
        for k, d, text in prompts:
            ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to(ctx.device)
            with torch.no_grad():
                out = model(ids, output_hidden_states=True)
            H.append(out.hidden_states[L][0, -1].float().cpu().numpy())
            P.append(out.logits[0, -1, day_ids].float().softmax(-1).cpu().numpy())
            ans.append((d + k) % 7)
        H, P, ans = np.stack(H), np.stack(P), np.array(ans)
        acc = float((P.argmax(1) == ans).mean())
        mu = H.mean(0)
        U, S, Vt = np.linalg.svd(H - mu, full_matrices=False)
        B = Vt[: min(64, len(H) - 1)]
        Z = (H - mu) @ B.T
        C = np.stack([Z[ans == c].mean(0) for c in range(7)])  # activation centroids per answer day
        Pc = np.stack([P[ans == c].mean(0) for c in range(7)])
        act_spline = periodic_spline(C)
        beh_spline = periodic_spline(np.sqrt(Pc))
        beh_dense = beh_spline(np.linspace(0, 7, 400))
        beh_dense = beh_dense / np.linalg.norm(beh_dense, axis=1, keepdims=True)

        def bc_dist(p):
            sq = np.sqrt(np.clip(p, 1e-12, 1))
            return float(-np.log(np.clip(beh_dense @ sq, 1e-12, 1)).min())

        def run_path(src_idx: int, path_z: np.ndarray) -> np.ndarray:
            text = prompts[src_idx][2]
            ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to(ctx.device)
            h0 = torch.tensor(H[src_idx], device=ctx.device)
            Bt = torch.tensor(B, device=ctx.device, dtype=torch.float32)
            mut = torch.tensor(mu, device=ctx.device, dtype=torch.float32)
            outs = []
            for z in path_z:
                target = h0 - Bt.T @ (Bt @ (h0 - mut)) + Bt.T @ torch.tensor(z, device=ctx.device, dtype=torch.float32)

                def fn(m, i, o, target=target):
                    hs = o[0] if isinstance(o, tuple) else o
                    hs[0, -1] = target.to(hs.dtype)
                    return o

                hk = model.model.layers[L - 1].register_forward_hook(fn)
                with torch.no_grad():
                    outs.append(model(ids).logits[0, -1, day_ids].float().softmax(-1).cpu().numpy())
                hk.remove()
            return np.stack(outs)

        e_man, e_lin, mid_man, mid_lin, examples = [], [], [], [], []
        for i in range(7):
            for step in (2, 3, -2, -3):
                j = (i + step) % 7
                src = next(n for n, (k, d, _) in enumerate(prompts) if ans[n] == i and P[n].argmax() == i) if any((ans == i) & (P.argmax(1) == i)) else None
                if src is None:
                    continue
                ts = np.linspace(0, 1, K)
                arc = act_spline(i + step * ts)  # shorter arc between the two days (step sign = direction)
                chord = (1 - ts)[:, None] * C[i] + ts[:, None] * C[j]
                pm, pl = run_path(src, arc), run_path(src, chord)
                inter = {(i + s) % 7 for s in range(1, abs(step))} if step > 0 else {(i - s) % 7 for s in range(1, abs(step))}
                e_man.append(np.mean([bc_dist(p) for p in pm]))
                e_lin.append(np.mean([bc_dist(p) for p in pl]))
                mid_man.append(np.mean([p.argmax() in inter for p in pm]))
                mid_lin.append(np.mean([p.argmax() in inter for p in pl]))
                if len(examples) < 6 and step in (3, -3):
                    examples.append({"from": i, "to": j, "manifold": pm, "linear": pl})
        res = {"layer": L, "accuracy": acc, "E_manifold": float(np.mean(e_man)), "E_linear": float(np.mean(e_lin)), "via_manifold": float(np.mean(mid_man)), "via_linear": float(np.mean(mid_lin)), "n_paths": len(e_man), "centroids_2d": C[:, :2], "examples": examples}
        results.append(res)
        log.info("manifold steering L%d: acc %.2f, E_BC manifold %.3f vs linear %.3f; via-intermediate %.2f vs %.2f (n=%d)", L, acc, res["E_manifold"], res["E_linear"], res["via_manifold"], res["via_linear"], len(e_man))
    best = min(results, key=lambda r: r["E_manifold"] / max(r["E_linear"], 1e-9))
    summary = {k: best[k] for k in ("layer", "accuracy", "E_manifold", "E_linear", "via_manifold", "via_linear", "n_paths")}
    repl = [
        verdict_row(
            "Steering along the manifold stays on the behaviour manifold",
            "weekdays, Llama-3.1-8B: E_BC 0.34 (manifold) vs 0.93 (linear)",
            f"Qwen3-1.7B layer {best['layer']}: E_BC {best['E_manifold']:.3f} (manifold) vs {best['E_linear']:.3f} (linear) over {best['n_paths']} paths",
            "manifold E_BC < 0.7 × linear E_BC",
            best["E_manifold"] < 0.7 * best["E_linear"],
            partial=best["E_manifold"] < best["E_linear"],
        ),
        verdict_row(
            "Manifold paths pass through the intermediate days",
            "manifold steering moves Mon → Tue → Wed; linear steering cuts across",
            f"fraction of path points predicting an intermediate day: {best['via_manifold'] * 100:.0f}% (manifold) vs {best['via_linear'] * 100:.0f}% (linear)",
            "manifold ≥ linear + 15 points",
            best["via_manifold"] >= best["via_linear"] + 0.15,
            partial=best["via_manifold"] > best["via_linear"],
        ),
    ]
    write_entry(SLUG, {"layers": results, "days": DAYS}, summary, repl, models=[MODEL], runtime_s=time.time() - t0, ndigits=3)
    del model
    torch.cuda.empty_cache()
