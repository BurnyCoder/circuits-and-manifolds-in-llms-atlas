"""Numbers live on a generalized helix (Kantamneni & Tegmark 2025) — tested across the Pythia family (160M → 2.8B).

Paper: "Language Models Use Trigonometry to Do Addition" (https://arxiv.org/abs/2502.00873). In GPT-J 6B, Pythia-6.9B
and Llama-3.1-8B, the residual stream at a number token a is well described by
    helix(a) = C · B(a),   B(a) = [a, cos(2πa/T), sin(2πa/T) for T in (2, 5, 10, 100)]  (9 dims),
fitted by linear regression after projecting activations to 100 PCA dims; activation patching along the fitted helix
recovers most of the effect of patching the whole layer (logit difference 7.21 vs 8.34 full; PCA 6.13).
No replication below 6B was found in our literature review — this entry tests the five smaller Pythia models
(https://github.com/EleutherAI/pythia), whose GPT-NeoX tokenizer has single tokens for 0–99 like GPT-J's.
Per model we report: Fourier spectrum of the number-token embeddings (Zhou et al. 2024 found periods 2, 2.5, 5, 10),
in-sample and held-out (80/20 split over a) helix R² per layer vs a shuffled-a control and simpler bases, and
activation patching (clean a+b vs corrupted a'+b, a+b < 100 so answers are single tokens) for full-layer, helix,
circle-only, linear-only and PCA-9 patches.
"""

from __future__ import annotations

import time

import numpy as np
import torch

from ..common.io import verdict_row, write_entry
from ..common.log import get_logger
from ..common.stats import bootstrap_ci

log = get_logger()
SLUG = "number-helix"
PERIODS = (2, 5, 10, 100)
MODELS = ["EleutherAI/pythia-160m", "EleutherAI/pythia-410m", "EleutherAI/pythia-1b", "EleutherAI/pythia-1.4b", "EleutherAI/pythia-2.8b"]
N_RAND9 = 5  # random 9-D subspaces per prompt (pre-stated control, review follow-ups 2026-11-28)
MIN_PATCH_N = 20  # minimum correctly solved prompts for a patching verdict (amendment 2026-11-27, before the full run)
FEWSHOT = ["12+7=19\n31+22=53\n40+8=48\n", "5+11=16\n27+30=57\n63+14=77\n", "8+8=16\n44+21=65\n19+60=79\n"]


def basis(a: np.ndarray, periods=PERIODS, linear=True) -> np.ndarray:
    """[n, 1 + 2*len(periods)] helix basis (a is NOT normalized, as in the paper)."""
    cols = [a.astype(float)] if linear else []
    for T in periods:
        cols += [np.cos(2 * np.pi * a / T), np.sin(2 * np.pi * a / T)]
    return np.stack(cols, 1)


def fit_r2(Z: np.ndarray, B: np.ndarray, tr: np.ndarray, te: np.ndarray) -> tuple[float, float, np.ndarray]:
    """Least-squares Z ≈ [B, 1] W on train rows; return (in-sample R², held-out R², W)."""
    A = np.c_[B, np.ones(len(B))]
    W, *_ = np.linalg.lstsq(A[tr], Z[tr], rcond=None)

    def r2(rows):
        res = ((Z[rows] - A[rows] @ W) ** 2).sum()
        tot = ((Z[rows] - Z[tr].mean(0)) ** 2).sum()
        return 1 - res / tot

    return float(r2(tr)), float(r2(te)), W


@torch.no_grad()
def run_model(name: str, ctx) -> dict:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name)
    dtype = torch.float16 if "2.8b" in name or "1.4b" in name else torch.float32
    model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype).to(ctx.device).eval()
    nl = model.config.num_hidden_layers
    num_ids = [tok(str(a), add_special_tokens=False).input_ids for a in range(100)]
    single = [len(x) == 1 for x in num_ids]
    assert all(single), f"{name}: numbers 0-99 are not all single tokens"
    num_ids = [x[0] for x in num_ids]

    # 1) Fourier spectrum of number-token embeddings over a = 0..99 (magnitude per period, averaged over dims).
    E = model.get_input_embeddings().weight[num_ids].float().cpu().numpy()  # [100, d]
    F = np.abs(np.fft.rfft(E - E.mean(0), axis=0)).mean(1)  # [51]
    periods = 100 / np.arange(1, 51)
    spectrum = {"period": periods, "mag": F[1:]}

    # 2) Activations at token a (prefix = few-shot examples + "a"); the token's state does not depend on b (causal LM).
    layers_out = []

    def acts_for(prefix: str) -> np.ndarray:
        X = []
        for a in range(100):
            ids = tok(prefix + str(a), return_tensors="pt", add_special_tokens=False).input_ids.to(ctx.device)
            hs = model(ids, output_hidden_states=True).hidden_states  # tuple of (nl+1) [1, pos, d]
            X.append(torch.stack([h[0, -1] for h in hs]).float().cpu().numpy())
        return np.stack(X)  # [100, nl+1, d]

    Xall = np.mean([acts_for(p) for p in (FEWSHOT if not ctx.quick else FEWSHOT[:1])], 0)
    a = np.arange(100)
    rng = np.random.default_rng(0)
    perm = rng.permutation(100)
    tr, te = perm[:80], perm[80:]
    shuffled = rng.permutation(100)
    bases = {
        "helix": basis(a),
        "circles only (no linear term)": basis(a, linear=False),
        "linear only": a[:, None].astype(float),
        "shuffled helix (control)": basis(shuffled),
    }
    folds = np.array_split(rng.permutation(100), 5)  # 5-fold CV over a (review follow-ups; was one 80/20 split)
    for L in range(nl + 1):
        X = Xall[:, L]
        Xc = X - X.mean(0)
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        Z = Xc @ Vt[:99].T  # all 99 non-trivial PCA dims (n = 100)
        row = {"layer": L}
        for k, B in bases.items():
            r_in, r_te, _ = fit_r2(Z, B, tr, te)
            cv = [fit_r2(Z, B, np.setdiff1d(a, f), f)[1] for f in folds]
            row[k] = {"in": r_in, "test": r_te, "cv": float(np.mean(cv))}
        layers_out.append(row)
    best_L = int(np.argmax([r["helix"]["test"] for r in layers_out[1:]])) + 1

    # Circle coordinates at the best layer: project activations on the fitted cos/sin directions of each period.
    X = Xall[:, best_L]
    Xc = X - X.mean(0)
    A = np.c_[basis(a), np.ones(100)]
    W, *_ = np.linalg.lstsq(A, Xc, rcond=None)  # [10, d] helix columns in residual space
    circles = {}
    for i, T in enumerate(PERIODS):
        u, v = W[1 + 2 * i], W[2 + 2 * i]
        circles[str(T)] = np.stack([Xc @ u / (u @ u), Xc @ v / (v @ v)], 1)
    linear_coord = Xc @ W[0] / (W[0] @ W[0])

    # 3) Accuracy and activation patching on a+b (a+b < 100). Clean a vs corrupted a' (same b).
    pairs = []
    for _ in range(60 if ctx.quick else 200):
        a1 = int(rng.integers(0, 90))
        b = int(rng.integers(0, 99 - a1))
        a2 = int(rng.integers(0, 99 - b))
        if a2 != a1:
            pairs.append((a1, a2, b))
    prefix = FEWSHOT[0]

    def prompt(a_, b_):
        return prefix + f"{a_}+{b_}="

    correct = []
    for a1, a2, b in pairs:
        ids = tok(prompt(a1, b), return_tensors="pt", add_special_tokens=False).input_ids.to(ctx.device)
        correct.append(int(model(ids).logits[0, -1].argmax()) == num_ids[a1 + b])
    acc = float(np.mean(correct))
    use = [p for p, c in zip(pairs, correct) if c]
    a_pos = len(tok(prefix, add_special_tokens=False).input_ids)  # index of the a token in the prompt

    patch_layers = sorted({1, nl // 4, nl // 2, (3 * nl) // 4})
    patch = []
    for L in patch_layers:
        X = Xall[:, L]
        mu = X.mean(0)
        Xc = X - mu
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        P9 = Vt[:9]
        rng9 = np.random.default_rng(1000 + L)
        span = Vt[:99]  # the 99-D span of the number-token activations at this layer
        Q9 = [np.linalg.qr((rng9.normal(size=(9, 99)) @ span).T)[0].T for _ in range(N_RAND9)]  # random orthonormal 9-D bases in it
        fits = {}
        for k in ("helix", "circles only (no linear term)", "linear only"):
            B = bases[k]
            A = np.c_[B, np.ones(100)]
            fits[k] = (np.linalg.lstsq(A, Xc, rcond=None)[0], B)
        rec = {k: [] for k in ("full layer", "helix", "circles only (no linear term)", "linear only", "PCA-9", "random 9-D")}
        for a1, a2, b in use[: (20 if ctx.quick else 80)]:
            clean_ids = tok(prompt(a1, b), return_tensors="pt", add_special_tokens=False).input_ids.to(ctx.device)
            corr_ids = tok(prompt(a2, b), return_tensors="pt", add_special_tokens=False).input_ids.to(ctx.device)
            assert int(clean_ids[0, a_pos]) == num_ids[a1] and int(corr_ids[0, a_pos]) == num_ids[a2], "a-token position mismatch"
            hc = model(clean_ids, output_hidden_states=True).hidden_states[L][0, a_pos].float()
            hx = model(corr_ids, output_hidden_states=True).hidden_states[L][0, a_pos].float()
            ld = lambda lg: (lg[num_ids[a1 + b]] - lg[num_ids[a2 + b]]).item()  # noqa: E731
            ld_clean = ld(model(clean_ids).logits[0, -1].float())
            ld_corr = ld(model(corr_ids).logits[0, -1].float())
            deltas = {"full layer": (hc - hx).cpu().numpy()}
            for k, (Wk, B) in fits.items():
                deltas[k] = (B[a1] - B[a2]) @ Wk[: B.shape[1]]  # move along the fitted manifold from a' to a
            deltas["PCA-9"] = P9.T @ (P9 @ (hc - hx).cpu().numpy())
            deltas["random 9-D"] = np.stack([Q.T @ (Q @ (hc - hx).cpu().numpy()) for Q in Q9])  # [N_RAND9, d], one batched pass
            for k, dlt in deltas.items():
                # Hidden state L = output of block L-1 (hidden_states[0] = embeddings) -> hook block L-1.
                dvec = torch.tensor(dlt, device=ctx.device)

                def hook(m, i, o, dvec=dvec):
                    hs = o[0] if isinstance(o, tuple) else o
                    hs[:, a_pos] += dvec.to(hs.dtype)  # [d] or, for the random 9-D draws, [N_RAND9, d] (one per batch row)
                    return o

                h = model.gpt_neox.layers[L - 1].register_forward_hook(hook)  # patch_layers are all >= 1
                ids_k = corr_ids.repeat(len(dvec), 1) if dvec.ndim == 2 else corr_ids
                lps = [ld(lg.float()) for lg in model(ids_k).logits[:, -1]]
                h.remove()
                rec[k].append(float(np.mean([(lp - ld_corr) / (ld_clean - ld_corr + 1e-9) for lp in lps])))
        patch.append({"layer": L, **{k: float(np.median(v)) if v else None for k, v in rec.items()}, "n": len(rec["helix"]),
                      "per_prompt": {k: np.array(v) for k, v in rec.items()}})
        log.info("%s L%d patch recovery: %s", name, L, {k: round(v, 3) if isinstance(v, float) else v for k, v in patch[-1].items()})
    del model
    torch.cuda.empty_cache()
    return {
        "name": name,
        "n_layers": nl,
        "spectrum": spectrum,
        "layers": layers_out,
        "best_layer": best_L,
        "circles": circles,
        "linear": linear_coord,
        "accuracy": acc,
        "n_pairs_correct": len(use),
        "patch": patch,
    }


def compute(ctx) -> None:
    t0 = time.time()
    models = MODELS[:2] if ctx.quick else MODELS
    res = [run_model(m, ctx) for m in models]
    for r in res:
        b = r["layers"][r["best_layer"]]
        log.info("%s: best layer %d helix test R² %.3f (shuffled %.3f), acc %.2f", r["name"], r["best_layer"], b["helix"]["test"], b["shuffled helix (control)"]["test"], r["accuracy"])
    big = res[-1]
    bl = big["layers"][big["best_layer"]]
    best_patch = max(big["patch"], key=lambda p: p["helix"] or -1)
    # Spectrum peaks: top-3 periods by magnitude in the largest model.
    top_periods = [float(big["spectrum"]["period"][i]) for i in np.argsort(-big["spectrum"]["mag"])[:4]]
    summary = {
        "largest": big["name"].split("/")[-1],
        "best_layer": big["best_layer"],
        "helix_test_r2": bl["helix"]["test"],
        "shuffled_test_r2": bl["shuffled helix (control)"]["test"],
        "linear_test_r2": bl["linear only"]["test"],
        "circles_test_r2": bl["circles only (no linear term)"]["test"],
        "acc_largest": big["accuracy"],
        "patch_layer": best_patch["layer"],
        "patch_helix": best_patch["helix"] or 0.0,
        "patch_full": best_patch["full layer"] or 0.0,
        "patch_pca9": best_patch["PCA-9"] or 0.0,
        "patch_linear": best_patch["linear only"] or 0.0,
        "patch_n": best_patch["n"],
        "top_periods": ", ".join(f"{p:.3g}" for p in top_periods),
        "n_models": len(res),
    }
    # Row 2 (control fix): 5-fold CV over a, max over layers for every basis including the shuffled control.
    cvmax = {k: max(r[k]["cv"] for r in big["layers"][1:]) for k in ("helix", "shuffled helix (control)", "linear only", "circles only (no linear term)")}
    cv_layer = 1 + int(np.argmax([r["helix"]["cv"] for r in big["layers"][1:]]))
    summary.update({"helix_cv_r2": cvmax["helix"], "shuffled_cv_r2": cvmax["shuffled helix (control)"], "linear_cv_r2": cvmax["linear only"],
                    "circles_cv_r2": cvmax["circles only (no linear term)"], "cv_layer": cv_layer})
    # Pre-stated test (review follow-ups): helix vs random 9-D and PCA-9 at every layer where the full-layer patch
    # recovers ≥ 50%; paired per-prompt differences, bootstrap CI of the mean.
    qual = [p for p in big["patch"] if (p["full layer"] or 0) >= 0.5]
    spec = []
    for p in qual:
        pp = p["per_prompt"]
        spec.append({"layer": p["layer"], "vs_random9": bootstrap_ci(pp["helix"] - pp["random 9-D"]), "vs_pca9": bootstrap_ci(pp["helix"] - pp["PCA-9"]),
                     "helix": float(np.mean(pp["helix"])), "random9": float(np.mean(pp["random 9-D"])), "pca9": float(np.mean(pp["PCA-9"]))})
        log.info("%s L%d helix − random-9 %s, helix − PCA-9 %s (mean recovery helix %.2f, random-9 %.2f, PCA-9 %.2f)", big["name"], p["layer"],
                 np.round(spec[-1]["vs_random9"], 3), np.round(spec[-1]["vs_pca9"], 3), spec[-1]["helix"], spec[-1]["random9"], spec[-1]["pca9"])
    beats_r9 = bool(spec) and all(x["vs_random9"][1] > 0 for x in spec)
    beats_p9 = bool(spec) and all(x["vs_pca9"][1] > 0 for x in spec)
    summary["patch_spec"] = "; ".join(
        f"L{x['layer']}: helix {x['helix'] * 100:.0f}% vs random 9-D {x['random9'] * 100:.0f}% (diff {x['vs_random9'][0] * 100:+.0f} [{x['vs_random9'][1] * 100:+.0f}, {x['vs_random9'][2] * 100:+.0f}]) "
        f"vs PCA-9 {x['pca9'] * 100:.0f}% (diff {x['vs_pca9'][0] * 100:+.0f} [{x['vs_pca9'][1] * 100:+.0f}, {x['vs_pca9'][2] * 100:+.0f}])" for x in spec) or "no layer with full-layer recovery ≥ 50%"
    for p in big["patch"] + [q for r in res[:-1] for q in r["patch"]]:
        p.pop("per_prompt", None)  # raw per-prompt arrays are only needed for the test above
    repl = [
        verdict_row(
            "Number embeddings are periodic with periods 2, 5, 10",
            "Fourier peaks at T = 2, 5, 10 in GPT-J (Kantamneni & Tegmark Fig. 2); 2, 2.5, 5, 10 (Zhou et al. 2024)",
            f"{summary['largest']}: strongest embedding periods {summary['top_periods']}",
            "at least two of {2, 5, 10} among the top-4 periods (±5%)",
            sum(any(abs(p - T) / T < 0.05 for p in top_periods) for T in (2, 5, 10)) >= 2,
        ),
        verdict_row(
            "Numbers lie on a generalized helix (T = 2, 5, 10, 100)",
            "helix fits the residual stream of GPT-J / Pythia-6.9B / Llama-3.1-8B",
            f"{summary['largest']}: 5-fold CV R² over a, max over layers for each basis: helix {cvmax['helix']:.2f} (layer {cv_layer}) vs "
            f"{cvmax['shuffled helix (control)']:.2f} for a shuffled helix, {cvmax['linear only']:.2f} linear-only, {cvmax['circles only (no linear term)']:.2f} circles only",
            "held-out helix R² ≥ 0.5 and ≥ 0.2 above the shuffled control (control fix 2026-11-28: CV, controls max-selected too)",
            cvmax["helix"] >= 0.5 and cvmax["helix"] - cvmax["shuffled helix (control)"] >= 0.2,
            partial=cvmax["helix"] - cvmax["shuffled helix (control)"] >= 0.1,
        ),
        verdict_row(
            "Patching along the helix recovers most of the full-layer effect",
            "GPT-J: helix 7.21 vs full 8.34 logit diff (86%); PCA 6.13",
            f"{summary['largest']} layer {best_patch['layer']}: median recovery helix {summary['patch_helix'] * 100:.0f}%, full layer {summary['patch_full'] * 100:.0f}%, "
            f"PCA-9 {summary['patch_pca9'] * 100:.0f}%, linear-only {summary['patch_linear'] * 100:.0f}% (n = {best_patch['n']} correct prompts; accuracy {big['accuracy'] * 100:.0f}%)",
            f"helix ≥ 50% of the full-layer recovery and ≥ linear-only; full-layer recovery ≥ 50% (m7 guard, 2026-11-28) (needs ≥ {MIN_PATCH_N} correctly solved prompts, else n/a)",
            # Too few solved prompts -> no verdict (small Pythia models rarely solve a+b; see planned-runs amendment).
            None
            if best_patch["n"] < MIN_PATCH_N
            else summary["patch_full"] >= 0.5 and summary["patch_helix"] >= 0.5 * summary["patch_full"] and summary["patch_helix"] >= summary["patch_linear"],
            partial=summary["patch_helix"] > summary["patch_linear"],
        ),
        verdict_row(
            "Confirmatory: patching along the helix beats generic 9-D subspaces",
            "GPT-J: helix 7.21 vs PCA 6.13 logit difference (Kantamneni & Tegmark Table 1)",
            f"{summary['largest']}, mean per-prompt recovery, paired differences [95% bootstrap CI]: {summary['patch_spec']}",
            "CI > 0 against random 9-D and PCA-9 at every layer with full-layer recovery ≥ 50%; partial if only against random 9-D (pre-stated 2026-11-28)",
            None if best_patch["n"] < MIN_PATCH_N or not spec else beats_r9 and beats_p9,
            partial=beats_r9,
        ),
    ]
    write_entry(SLUG, {"models": res, "periods": PERIODS}, summary, repl, models=[r["name"] for r in res], runtime_s=time.time() - t0, ndigits=3)
    log.info("number helix summary: %s", summary)
