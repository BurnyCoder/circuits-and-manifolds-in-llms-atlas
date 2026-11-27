"""Fourier probes for the "geometric calculator" (Feucht et al. 2026), across small Pythia models.

Paper: Feucht, Haklay, Bhalla, Wurgaft, et al., "Arithmetic in the Wild: Llama uses Base-10 Addition to Reason About
Cyclic Concepts" (https://arxiv.org/abs/2605.01148, CC BY 4.0; Goodfire post
https://www.goodfire.com/research/a-geometric-calculator). In Llama-3.1-8B, affine probes on the last token of
"{a}+{b}=" recover sin/cos(2π(a+b)/T) best for periods T ∈ {2, 5, 10, 20, 50, 100} (base-10 circles), computed by 28
layer-18 MLP neurons that are shared with month/weekday arithmetic.
The paper reports no small-model evidence. Here we fit the same probes (ridge, held-out pairs) for every period
T = 2…150 on the last token of "{a}+{b}=" (a, b ∈ 1…99) in Pythia-410M, -1.4B and -2.8B at several layers, and check
whether the base-10 periods stand out.
"""

from __future__ import annotations

import time

import numpy as np
import torch

from ..common.io import verdict_row, write_entry
from ..common.log import get_logger

log = get_logger()
SLUG = "calculator-probes"
MODELS = ["EleutherAI/pythia-410m", "EleutherAI/pythia-1.4b", "EleutherAI/pythia-2.8b"]
PERIODS = np.arange(2, 151)
BASE10 = [2, 5, 10, 20, 50, 100]


ALPHAS = [1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5]


def probe_r2(X: np.ndarray, s: np.ndarray, tr: np.ndarray, alpha: float | None = None) -> np.ndarray:
    """Held-out R² of ridge probes X -> [sin, cos](2π s / T) for every period T (averaged over sin and cos).

    alpha=None: ridge strength chosen *per period* from ALPHAS on an inner split of the training rows (every 5th), as
    sklearn's RidgeCV(alpha_per_target=True); the test rows are never used for the choice. (One alpha for all periods,
    picked by mean R², fails: most periods are unpredictable and reward maximal shrinkage, which also flattens the
    real circles.)"""
    if alpha is None:
        idx = np.where(tr)[0]
        inner = np.zeros(len(s), bool)
        inner[idx[::5]] = True
        keep = tr  # inner split uses training rows only: fit on tr & ~inner, validate on inner
        val = np.stack([_probe(X[keep], s[keep], (tr & ~inner)[keep], a) for a in ALPHAS])  # [n_alpha, n_periods]
        test = np.stack([_probe(X, s, tr, a) for a in ALPHAS])
        return test[val.argmax(0), np.arange(len(PERIODS))]
    return _probe(X, s, tr, alpha)


def _probe(X: np.ndarray, s: np.ndarray, tr: np.ndarray, alpha: float) -> np.ndarray:
    """Held-out (rows ~tr) R² for every period with a fixed ridge strength; features standardized on rows tr."""
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
    A = (X - mu) / sd
    G = A[tr].T @ A[tr] + alpha * np.eye(A.shape[1])
    Ginv_At = np.linalg.solve(G, A[tr].T)  # shared across all targets
    out = []
    for T in PERIODS:
        Y = np.c_[np.sin(2 * np.pi * s / T), np.cos(2 * np.pi * s / T)]
        W = Ginv_At @ (Y[tr] - Y[tr].mean(0))
        pred = A[~tr] @ W + Y[tr].mean(0)
        r2 = 1 - ((Y[~tr] - pred) ** 2).sum(0) / ((Y[~tr] - Y[~tr].mean(0)) ** 2).sum(0)
        out.append(float(r2.mean()))
    return np.array(out)


def peak_scores(r2: np.ndarray) -> np.ndarray:
    """How much each period stands out from its neighbours: R²(T) − mean R²(T±1) (edge periods use one neighbour).
    Smooth trends (e.g. long periods acting as a monotone magnitude code) score ~0; isolated peaks score high."""
    left = np.r_[r2[1], r2[:-1]]
    right = np.r_[r2[1:], r2[-2]]
    return r2 - (left + right) / 2


def compute(ctx) -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.time()
    rng = np.random.default_rng(ctx.seed)
    n = 400 if ctx.quick else 1500
    pairs = rng.integers(1, 100, size=(n, 2))
    s = pairs.sum(1)
    tr = rng.random(n) < 0.8
    res = []
    for name in MODELS[:1] if ctx.quick else MODELS:
        tok = AutoTokenizer.from_pretrained(name)
        model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float16).to(ctx.device).eval()
        nl = model.config.num_hidden_layers
        layers = sorted({nl // 4, nl // 2, (2 * nl) // 3, (3 * nl) // 4})
        H = {L: [] for L in layers}
        for i in range(0, n, 64):
            texts = [f"{a}+{b}=" for a, b in pairs[i : i + 64]]
            tok.padding_side = "left"
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            b_ = tok(texts, return_tensors="pt", padding=True).to(ctx.device)
            with torch.no_grad():
                hs = model(**b_, output_hidden_states=True).hidden_states
            for L in layers:
                H[L].append(hs[L][:, -1].float().cpu().numpy())
        per_layer = []
        for L in layers:
            r2 = probe_r2(np.concatenate(H[L]), s, tr)
            top = PERIODS[np.argsort(-r2)[:8]].tolist()
            top_peak = PERIODS[np.argsort(-peak_scores(r2))[:8]].tolist()
            per_layer.append({"layer": L, "r2": r2, "top": top, "top_peak": top_peak})
            log.info("%s L%d: top periods by peak score %s", name, L, top_peak)
            log.info("%s L%d: top periods %s (R² %s)", name, L, top[:6], np.round(np.sort(r2)[::-1][:6], 3))
        best = max(per_layer, key=lambda x: sum(x["r2"][PERIODS == T][0] for T in (2, 5, 10)))
        res.append({"name": name, "n_layers": nl, "layers": per_layer, "best_layer": best["layer"], "top": best["top"]})
        del model
        torch.cuda.empty_cache()
    big = res[-1]
    best = next(x for x in big["layers"] if x["layer"] == big["best_layer"])
    hits = [T for T in BASE10 if T in best["top"][:8]]
    hits_peak = [T for T in BASE10 if T in best["top_peak"][:8]]
    r2_at = {T: float(best["r2"][PERIODS == T][0]) for T in BASE10}
    neigh = {T: float(np.mean([best["r2"][PERIODS == u][0] for u in (T - 1, T + 1) if u in PERIODS])) for T in (5, 10, 20)}
    summary = {
        "model": big["name"].split("/")[-1],
        "layer": big["best_layer"],
        "top_periods": ", ".join(map(str, best["top"][:8])),
        "n_base10_in_top8": len(hits),
        "top_peak_periods": ", ".join(map(str, best["top_peak"][:8])),
        "n_base10_in_top8_peaks": len(hits_peak),
        "r2_T10": r2_at[10],
        "r2_T10_neighbours": neigh[10],
        "r2_T5": r2_at[5],
        "r2_T2": r2_at[2],
    }
    repl = [
        verdict_row(
            "Addition results are encoded on base-10 circles (T = 2, 5, 10, …)",
            "Fourier probes peak at T ∈ {2, 5, 10, 20, 50, 100} in Llama-3.1-8B (Feucht et al.)",
            f"{summary['model']} layer {big['best_layer']}: top-8 periods {summary['top_periods']} ({len(hits)} of the base-10 set); R²(T=10) {r2_at[10]:.2f} vs neighbours {neigh[10]:.2f}",
            "≥ 3 base-10 periods in the top 8 and R²(T=10) ≥ neighbours + 0.1",
            len(hits) >= 3 and r2_at[10] >= neigh[10] + 0.1,
            partial=len(hits) >= 2,
        ),
        verdict_row(
            "Base-10 periods stand out as peaks (amendment: peak score, removes the long-period magnitude trend)",
            "Fourier probes peak at T ∈ {2, 5, 10, 20, 50, 100} in Llama-3.1-8B (Feucht et al.)",
            f"{summary['model']} layer {big['best_layer']}: top-8 periods by R²(T) − mean R²(T±1): {summary['top_peak_periods']} ({len(hits_peak)} of the base-10 set)",
            "≥ 3 base-10 periods among the top-8 peaks and R²(T=10) ≥ neighbours + 0.1 (same thresholds)",
            len(hits_peak) >= 3 and r2_at[10] >= neigh[10] + 0.1,
            partial=len(hits_peak) >= 2,
        ),
    ]
    write_entry(SLUG, {"models": res, "periods": PERIODS, "base10": BASE10}, summary, repl, models=MODELS, runtime_s=time.time() - t0, ndigits=3)
