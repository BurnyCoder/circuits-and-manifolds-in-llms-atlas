"""Steering along the manifold vs. straight lines (Wurgaft et al. 2026), on Qwen3-1.7B weekday arithmetic.

Paper: Wurgaft, Rager, Kowal, et al., "Manifold Steering Reveals the Shared Geometry of Neural Network Representation and
Behavior" (https://arxiv.org/abs/2605.05115; Goodfire post https://www.goodfire.ai/research/manifold-steering). In
Llama-3.1-8B (layer 28, 64-d PCA) they fit a periodic cubic spline through the activation centroids of the 7 weekday
answers and steer from one day to another either along the spline or along the straight chord. Along the manifold the
output distribution moves through the intermediate days and stays near the behaviour manifold (Bhattacharyya energy
E_BC 0.34 vs 0.93 for linear on weekdays; isometry r .99 vs .89).
Here (layers 22/24/26 of 28, where an endpoint check shows the answer-day subspace controls the output): day-arithmetic
prompts Qwen3-1.7B solves ("If today is {D}, then two days later it is"; calendar_circles.day_prompts;
the original "N days from D" template was answered correctly only 0-57% of the time), last-token activations at a middle
layer, the 6-d subspace spanned by the answer-day centroids (instead of the paper's 64-d PCA, which needs many more
prompts), periodic spline (scipy CubicSpline, bc_type='periodic') through the 7 centroids of *correctly answered*
prompts grouped by answer day (so each centroid is what the model represents when it says that day). For every
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
from ..common.stats import group_bootstrap
from .calendar_circles import DAYS, day_prompts

log = get_logger()
SLUG = "manifold-steering"
MODEL = "Qwen/Qwen3-1.7B"
ENDPOINT_ONLY = [12, 16, 20]  # earlier layers: endpoint check only (documents why steering uses 22–26)
LAYERS = [22, 24, 26]  # ~79–93% depth; the answer-day subspace is causal only from layer 22 on (endpoint check, amendment)
K = 13  # points per path
N_SRC = 4  # correctly answered source prompts per start day (review follow-ups: was 1)


def polyline(C: np.ndarray, i: int, step: int, ts: np.ndarray) -> np.ndarray:
    """Piecewise-linear path from centroid i through every intermediate day centroid to i + step (mod 7), reaching the
    m-th intermediate centroid at t = m/|step| like the periodic spline does (pre-stated control, 2026-11-28)."""
    sgn, n = int(np.sign(step)), abs(step)
    V = np.stack([C[(i + sgn * m) % 7] for m in range(n + 1)])
    u = n * ts
    seg = np.minimum(u.astype(int), n - 1)
    frac = (u - seg)[:, None]
    return (1 - frac) * V[seg] + frac * V[seg + 1]


def periodic_spline(points: np.ndarray) -> CubicSpline:
    t = np.arange(len(points) + 1)
    return CubicSpline(t, np.vstack([points, points[:1]]), bc_type="periodic")


def compute(ctx) -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(ctx.device).eval()
    day_ids = [tok(d, add_special_tokens=False).input_ids[0] for d in DAYS]
    prompts = day_prompts()
    results = []
    endpoint_only = []
    for L in (ENDPOINT_ONLY + LAYERS) if not ctx.quick else LAYERS[:1]:
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
        ok = P.argmax(1) == ans  # correctly answered prompts define the centroids and the steering sources
        # Steering subspace = span of the 7 answer-day centroids (6-d). The paper's 64-d PCA came from many prompts; on our
        # 63 it would span almost all prompt variation (prefix, offset), and overwriting it with a centroid creates
        # off-distribution hybrids. Replacing only the centroid-span coordinates keeps everything else in the prompt.
        Cfull = np.stack([H[ok & (ans == c)].mean(0) for c in range(7)])  # [7, d] centroids of correct prompts
        mu = Cfull.mean(0)
        B = np.linalg.svd(Cfull - mu, full_matrices=False)[2][:6]  # orthonormal basis [6, d]
        C = (Cfull - mu) @ B.T  # centroids in the 6-d day subspace
        Pc = np.stack([P[ok & (ans == c)].mean(0) for c in range(7)])
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

        # Endpoint check (the evidence behind choosing layers 22–26; see the amendment): writing day j's centroid into a
        # correctly answered prompt should make the model answer j if the day subspace is causal at this layer.
        ep_rng = np.random.default_rng(ctx.seed + L)
        ok_idx = np.where(ok)[0]
        ep = []
        for n in ep_rng.choice(ok_idx, min(28, len(ok_idx)), replace=False):
            j = int((ans[n] + ep_rng.integers(1, 7)) % 7)
            ep.append(int(run_path(int(n), C[j][None])[0].argmax()) == j)
        endpoint = float(np.mean(ep))
        log.info("manifold steering L%d: endpoint success %.2f (n=%d)", L, endpoint, len(ep))
        if L in ENDPOINT_ONLY:
            endpoint_only.append({"layer": L, "endpoint_success": endpoint})
            continue
        paths, examples = [], []
        for i in range(7):
            srcs = [n for n in range(len(prompts)) if ans[n] == i and P[n].argmax() == i][:N_SRC]
            for src in srcs:
                for step in (2, 3, -2, -3):
                    j = (i + step) % 7
                    ts = np.linspace(0, 1, K)
                    arc = act_spline(i + step * ts)  # shorter arc between the two days (step sign = direction)
                    chord = (1 - ts)[:, None] * C[i] + ts[:, None] * C[j]
                    pm, pl, pp = run_path(src, arc), run_path(src, chord), run_path(src, polyline(C, i, step, ts))
                    inter = {(i + np.sign(step) * s_) % 7 for s_ in range(1, abs(step))}
                    paths.append({"src": src, "E_man": np.mean([bc_dist(p) for p in pm]), "E_lin": np.mean([bc_dist(p) for p in pl]),
                                  "E_pl": np.mean([bc_dist(p) for p in pp]), "via_man": np.mean([p.argmax() in inter for p in pm]),
                                  "via_lin": np.mean([p.argmax() in inter for p in pl]), "via_pl": np.mean([p.argmax() in inter for p in pp])})
                    if len(examples) < 6 and step in (3, -3) and src == srcs[0]:
                        examples.append({"from": i, "to": j, "manifold": pm, "linear": pl, "polyline": pp})
        g = np.array([p["src"] for p in paths])
        col = {k: np.array([p[k] for p in paths]) for k in paths[0] if k != "src"}
        ci = lambda v, g=g: group_bootstrap(g, lambda idx: v[idx].mean(), seed=ctx.seed)  # noqa: E731  (cluster = source prompt)
        d_pl = col["E_man"] - col["E_pl"]
        res = {"layer": L, "accuracy": acc, "endpoint_success": endpoint, "n_paths": len(paths), "n_sources": int(len(np.unique(g))),
               "centroids_2d": C[:, :2], "examples": examples, "d_spline_pl": float(d_pl.mean()), "d_spline_pl_ci": ci(d_pl)}
        for k, v in col.items():
            res[{"E_man": "E_manifold", "E_lin": "E_linear", "E_pl": "E_polyline", "via_man": "via_manifold", "via_lin": "via_linear", "via_pl": "via_polyline"}[k]] = float(v.mean())
            res[k + "_ci"] = ci(v)
        results.append(res)
        log.info("manifold steering L%d: acc %.2f, E_BC manifold %.3f vs linear %.3f vs polyline %.3f; spline − polyline %.3f [%.3f, %.3f]; "
                 "via-intermediate %.2f / %.2f / %.2f (n=%d paths, %d sources)", L, acc, res["E_manifold"], res["E_linear"], res["E_polyline"],
                 res["d_spline_pl"], *res["d_spline_pl_ci"], res["via_manifold"], res["via_linear"], res["via_polyline"], len(paths), res["n_sources"])
    best = min(results, key=lambda r: r["E_manifold"] / max(r["E_linear"], 1e-9))
    summary = {k: best[k] for k in ("layer", "accuracy", "E_manifold", "E_linear", "via_manifold", "via_linear", "n_paths")}
    # Pre-stated test (review follow-ups): spline vs piecewise-linear through the same centroids, at every layer.
    n_below = sum(r["d_spline_pl_ci"][1] < 0 for r in results)
    summary["spline_vs_polyline"] = "; ".join(f"L{r['layer']} {r['d_spline_pl']:+.3f} [{r['d_spline_pl_ci'][0]:+.3f}, {r['d_spline_pl_ci'][1]:+.3f}]" for r in results)
    summary["n_layers_spline_better"] = n_below
    c = lambda k, pct=False: "[{:.0f}, {:.0f}]".format(*(100 * x for x in best[k])) if pct else "[{:.3f}, {:.3f}]".format(*best[k])  # noqa: E731
    repl = [
        verdict_row(
            "Steering along the manifold stays on the behaviour manifold",
            "weekdays, Llama-3.1-8B: E_BC 0.34 (manifold) vs 0.93 (linear)",
            f"Qwen3-1.7B layer {best['layer']}: E_BC {best['E_manifold']:.3f} {c('E_man_ci')} (manifold) vs {best['E_linear']:.3f} {c('E_lin_ci')} (linear) "
            f"over {best['n_paths']} paths from {best['n_sources']} source prompts (95% CIs over source prompts)",
            "manifold E_BC < 0.7 × linear E_BC",
            best["E_manifold"] < 0.7 * best["E_linear"],
            partial=best["E_manifold"] < best["E_linear"],
        ),
        verdict_row(
            "Manifold paths pass through the intermediate days",
            "manifold steering moves Mon → Tue → Wed; linear steering cuts across",
            f"fraction of path points predicting an intermediate day: {best['via_manifold'] * 100:.0f}% {c('via_man_ci', True)} (manifold) vs "
            f"{best['via_linear'] * 100:.0f}% {c('via_lin_ci', True)} (linear); piecewise-linear through the same centroids {best['via_polyline'] * 100:.0f}%",
            "manifold ≥ linear + 15 points",
            best["via_manifold"] >= best["via_linear"] + 0.15,
            partial=best["via_manifold"] > best["via_linear"],
        ),
        verdict_row(
            "Confirmatory: the smooth manifold path beats a piecewise-linear path through the same centroids",
            "the data-driven manifold geometry, not just visiting the right points, keeps behaviour on its manifold (Wurgaft et al.)",
            f"E_BC(spline) − E_BC(piecewise-linear), mean over paths [95% CI over source prompts]: {summary['spline_vs_polyline']}",
            "CI below 0 at all of layers 22, 24, 26; partial at 1–2 (pre-stated 2026-11-28)",
            n_below == len(results) == 3,
            partial=n_below >= 1,
        ),
    ]
    summary["endpoint_by_layer"] = ", ".join(f"L{e['layer']} {e['endpoint_success']:.2f}" for e in endpoint_only + [{"layer": r["layer"], "endpoint_success": r["endpoint_success"]} for r in results])
    write_entry(SLUG, {"layers": results, "endpoint_only": endpoint_only, "days": DAYS}, summary, repl, models=[MODEL], runtime_s=time.time() - t0, ndigits=3)
    del model
    torch.cuda.empty_cache()
