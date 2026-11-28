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
from ..common.stats import perm_pvalue

log = get_logger()
SLUG = "calculator-probes"
MODELS = ["EleutherAI/pythia-410m", "EleutherAI/pythia-1.4b", "EleutherAI/pythia-2.8b"]
PERIODS = np.arange(2, 151)
BASE10 = [2, 5, 10, 20, 50, 100]


ALPHAS = [1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5]


class ProbeBank:
    """Ridge probes X -> [sin, cos](2π s / T) for every period T, with the solves cached so that many label vectors
    (the real sums and permuted ones for the null) cost only matrix products.

    For each alpha: an inner fit on training rows outside ``inner`` (validated on ``inner``) and a fit on all training
    rows (tested on ~tr); features standardized on the fitting rows, targets centred on them. The alpha is chosen
    *per period* by inner-validation R², as sklearn's RidgeCV(alpha_per_target=True); test rows never enter the
    choice. (One alpha for all periods, picked by mean R², fails: most periods are unpredictable and reward maximal
    shrinkage, which also flattens the real circles.)"""

    def __init__(self, X: np.ndarray, tr: np.ndarray, inner: np.ndarray):
        self.fit_in, self.val, self.tr = tr & ~inner, inner, tr
        self.solves = [(self._solve(X, self.fit_in, a), self._solve(X, tr, a)) for a in ALPHAS]

    @staticmethod
    def _solve(X, rows, alpha):
        mu, sd = X[rows].mean(0), X[rows].std(0) + 1e-6
        A = (X - mu) / sd
        return np.linalg.solve(A[rows].T @ A[rows] + alpha * np.eye(A.shape[1]), A[rows].T), A

    @staticmethod
    def _r2(solve, Y, fit, ev):
        G, A = solve
        ym = Y[fit].mean(0)
        pred = A[ev] @ (G @ (Y[fit] - ym)) + ym
        r2 = 1 - ((Y[ev] - pred) ** 2).sum(0) / ((Y[ev] - Y[ev].mean(0)) ** 2).sum(0)
        return r2.reshape(len(PERIODS), 2).mean(1)  # average of the sin and cos targets per period

    def r2(self, s: np.ndarray) -> np.ndarray:
        """Held-out R² per period for the sums s (per-period alpha from the inner split)."""
        ang = 2 * np.pi * s[:, None] / PERIODS[None]
        Y = np.stack([np.sin(ang), np.cos(ang)], -1).reshape(len(s), -1)  # [n, 2·periods] (sin, cos) per period
        val = np.stack([self._r2(si, Y, self.fit_in, self.val) for si, _ in self.solves])
        test = np.stack([self._r2(st, Y, self.tr, ~self.tr) for _, st in self.solves])
        return test[val.argmax(0), np.arange(len(PERIODS))]


def inner_split(tr: np.ndarray, groups: np.ndarray | None = None) -> np.ndarray:
    """Every 5th training row (or every 5th training group, e.g. unordered pair) as the inner validation split."""
    if groups is None:
        inner = np.zeros(len(tr), bool)
        inner[np.where(tr)[0][::5]] = True
        return inner
    return tr & np.isin(groups, np.unique(groups[tr])[::5])


def probe_r2(X: np.ndarray, s: np.ndarray, tr: np.ndarray, groups: np.ndarray | None = None) -> np.ndarray:
    """Held-out R² of the Fourier probes for every period (see ProbeBank)."""
    return ProbeBank(X, tr, inner_split(tr, groups)).r2(s)


def valid_peak_hits(r2: np.ndarray, top: int = 8, floor: float = 0.05) -> list[int]:
    """Top periods by peak score among those with R²(T) > floor (pre-stated 2026-11-28: a near-zero R² is no peak)."""
    ok = r2 > floor
    order = [i for i in np.argsort(-peak_scores(r2)) if ok[i]][:top]
    return PERIODS[order].tolist()


def peak_scores(r2: np.ndarray) -> np.ndarray:
    """How much each period stands out from its neighbours: R²(T) − mean R²(T±1) (edge periods use one neighbour).
    Smooth trends (e.g. long periods acting as a monotone magnitude code) score ~0; isolated peaks score high."""
    left = np.r_[r2[1], r2[:-1]]
    right = np.r_[r2[1:], r2[-2]]
    return r2 - (left + right) / 2


def confirmatory(X: np.ndarray, s: np.ndarray, tr: np.ndarray, groups: np.ndarray, rng, n_perm: int) -> dict:
    """Pre-stated test (reports/planned-runs.md, review follow-ups): base-10 hits among the top-8 valid peaks at a fixed
    layer, a label-permutation null (sums shuffled across prompts, same probes) and a random-period-set null."""
    bank = ProbeBank(X, tr, inner_split(tr, groups))
    r2 = bank.r2(s)
    top = valid_peak_hits(r2)
    hits = [T for T in BASE10 if T in top]
    null_perm = [sum(T in valid_peak_hits(bank.r2(rng.permutation(s))) for T in BASE10) for _ in range(n_perm)]
    null_set = [len(set(rng.choice(PERIODS, len(BASE10), replace=False)) & set(top)) for _ in range(10000)]
    return {"r2": r2, "valid_top": top, "hits": hits, "p_perm": perm_pvalue(len(hits), null_perm), "p_set": perm_pvalue(len(hits), null_set),
            "null_perm_max": int(max(null_perm)), "r2_at": {str(T): float(r2[PERIODS == T][0]) for T in BASE10}}


def compute(ctx) -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.time()
    rng = np.random.default_rng(ctx.seed)
    n = 400 if ctx.quick else 1500
    pairs = rng.integers(1, 100, size=(n, 2))
    s = pairs.sum(1)
    tr = rng.random(n) < 0.8
    # Pre-stated confirmatory split (review follow-ups): 80/20 by unordered pair {a, b}, so neither a duplicate prompt nor
    # its commuted form b + a sits on both sides.
    key = pairs.min(1) * 100 + pairs.max(1)
    uk = np.unique(key)
    tr_g = ~np.isin(key, np.random.default_rng(ctx.seed + 7).choice(uk, int(0.2 * len(uk)), replace=False))
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
        conf = confirmatory(np.concatenate(H[(2 * nl) // 3]), s, tr_g, key, rng, 50 if ctx.quick else 200)
        conf["layer"] = (2 * nl) // 3
        log.info("%s confirmatory L%d (pair split, R² > 0.05): valid top-8 peaks %s, base-10 hits %s, p_perm %.4f, p_set %.4f",
                 name, conf["layer"], conf["valid_top"], conf["hits"], conf["p_perm"], conf["p_set"])
        res.append({"name": name, "n_layers": nl, "layers": per_layer, "best_layer": best["layer"], "top": best["top"], "confirmatory": conf})
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
    cf = big["confirmatory"]
    summary.update({
        "conf_layer": cf["layer"],
        "conf_valid_top": ", ".join(map(str, cf["valid_top"])) or "none",
        "conf_hits": ", ".join(map(str, cf["hits"])) or "none",
        "conf_n_hits": len(cf["hits"]),
        "conf_p_perm": cf["p_perm"],
        "conf_p_set": cf["p_set"],
        "conf_r2_T5": cf["r2_at"]["5"],
        "conf_r2_T10": cf["r2_at"]["10"],
        "conf_other_models": "; ".join(f"{r['name'].split('/')[-1]} L{r['confirmatory']['layer']}: hits {r['confirmatory']['hits'] or 'none'}" for r in res[:-1]),
    })
    sig = cf["p_perm"] < 0.05 and cf["p_set"] < 0.05
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
        verdict_row(
            "Confirmatory: base-10 periods stand out at a fixed layer with an R² floor and nulls",
            "Fourier probes peak at T ∈ {2, 5, 10, 20, 50, 100} in Llama-3.1-8B (Feucht et al.)",
            f"{summary['model']} layer {cf['layer']} (⌊2n/3⌋), split by unordered pair: top-8 peaks with R² > 0.05: {summary['conf_valid_top']}; "
            f"base-10 hits {summary['conf_hits']} (R²(5) {cf['r2_at']['5']:.2f}, R²(10) {cf['r2_at']['10']:.2f}); label-permutation p {cf['p_perm']:.3f}, "
            f"random-period-set p {cf['p_set']:.4f}. Other models: {summary['conf_other_models']}",
            "≥ 3 hits with both p < 0.05; partial with 2 hits (pre-stated 2026-11-28)",
            len(cf["hits"]) >= 3 and sig,
            partial=len(cf["hits"]) == 2 and sig,
        ),
    ]
    write_entry(SLUG, {"models": res, "periods": PERIODS, "base10": BASE10}, summary, repl, models=MODELS, runtime_s=time.time() - t0, ndigits=3)
