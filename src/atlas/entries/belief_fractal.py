"""Transformers linearly represent belief-state geometry — a fractal in the residual stream (trained here).

Replicates Shai, Marzen, Teixeira, Gietelink Oldenziel, Riechers 2024, "Transformers represent belief state geometry in
their residual stream" (https://arxiv.org/abs/2405.15943): a small transformer trained to predict tokens from the
Mess3 hidden Markov process (3 hidden states, 3 tokens; x = 0.05, α = 0.85, as in the paper) linearly embeds the
Bayes-optimal belief over hidden states — points on a 2-simplex that form a fractal — in its final residual stream.
Mess3 transition matrices T^(t)[i, j] = P(emit t, go to j | state i) follow Marzen & Crutchfield (2017) as used by the
paper: with β = (1 − α)/2 and y = 1 − 2x, the emitted token is "aligned" with the next state with weight α.
Beliefs update as η ← η T^(t) / (η T^(t) 1). We regress residual activations onto beliefs with linear least squares
(held-out sequences) across training checkpoints, against a random-init control.
"""

from __future__ import annotations

import time

import numpy as np
import torch

from ..common.env import RESULTS, seed_everything
from ..common.io import verdict_row, write_entry
from ..common.log import get_logger

log = get_logger()
SLUG = "belief-fractal"
CTX = 10


def mess3(x: float = 0.05, alpha: float = 0.85) -> np.ndarray:
    """[3 tokens, 3 states, 3 states] labelled transition matrices (rows sum to 1 over tokens and next states)."""
    b, y = (1 - alpha) / 2, 1 - 2 * x
    T = np.zeros((3, 3, 3))
    for t in range(3):
        for i in range(3):
            for j in range(3):
                emit = alpha if j == t else b  # token "points to" the next state
                stay = y if i == j else x
                T[t, i, j] = emit * stay
    assert np.allclose(T.sum((0, 2)), 1)
    return T


def sample(T: np.ndarray, n: int, rng) -> np.ndarray:
    """n sequences of CTX tokens from the HMM started at its stationary (uniform) distribution."""
    seqs = np.zeros((n, CTX), dtype=np.int64)
    state = rng.integers(0, 3, n)
    flat = T.transpose(1, 0, 2).reshape(3, 9)  # [state, token*3 + next]
    cum = flat.cumsum(1)
    for t in range(CTX):
        u = rng.random(n)
        k = (u[:, None] > cum[state]).sum(1)
        seqs[:, t], state = k // 3, k % 3
    return seqs


def beliefs(T: np.ndarray, seqs: np.ndarray) -> np.ndarray:
    """[n, CTX, 3] belief over hidden states after observing tokens 0..t (Bayesian filtering)."""
    n = len(seqs)
    eta = np.full((n, 3), 1 / 3)
    out = np.zeros((n, CTX, 3))
    for t in range(CTX):
        eta = np.einsum("ni,nij->nj", eta, T[seqs[:, t]])
        eta /= eta.sum(1, keepdims=True)
        out[:, t] = eta
    return out


def simplex_xy(b: np.ndarray) -> np.ndarray:
    """Barycentric -> 2-D triangle coordinates (vertices (0,0), (1,0), (0.5, √3/2))."""
    return np.stack([b[..., 1] + 0.5 * b[..., 2], b[..., 2] * np.sqrt(3) / 2], -1)


def entropy_rate(T: np.ndarray, rng, n: int = 20000) -> float:
    """Mean next-token entropy under the optimal predictor (lower bound on achievable loss), estimated by sampling."""
    seqs = sample(T, n, rng)
    b = beliefs(T, seqs)
    p = np.einsum("nti,kij->ntk", b[:, :-1], T).clip(1e-12)  # P(next token | belief)
    p /= p.sum(-1, keepdims=True)
    ll = np.log(np.take_along_axis(p, seqs[:, 1:, None], -1)[..., 0])
    return float(-ll.mean())


@torch.no_grad()
def regress(model, seqs, B, device, layer=None) -> dict:
    """Linear map from residual stream (final layer, or concat of all if layer='all') to beliefs; held-out R² + preds."""
    _, cache = model.run_with_cache(torch.tensor(seqs, device=device), names_filter=lambda n: "resid_post" in n)
    nl = model.cfg.n_layers
    X = torch.cat([cache["resid_post", l] for l in range(nl)], -1) if layer == "all" else cache["resid_post", nl - 1]
    X = X.reshape(-1, X.shape[-1]).cpu().numpy()
    Y = B.reshape(-1, 3)
    n = len(X)
    tr = np.arange(n) < int(0.8 * n)
    A = np.c_[X, np.ones(n)]
    W, *_ = np.linalg.lstsq(A[tr], Y[tr], rcond=None)
    P = A @ W
    r2 = 1 - ((Y[~tr] - P[~tr]) ** 2).sum() / ((Y[~tr] - Y[tr].mean(0)) ** 2).sum()
    mse = float(((Y[~tr] - P[~tr]) ** 2).mean())
    return {"r2": float(r2), "mse": mse, "pred": P}


def compute(ctx) -> None:
    from transformer_lens import HookedTransformer, HookedTransformerConfig

    t0 = time.time()
    seed_everything(ctx.seed)
    rng = np.random.default_rng(ctx.seed)
    T = mess3()
    cfg = HookedTransformerConfig(n_layers=4, d_model=64, n_heads=1, d_head=64, d_mlp=256, d_vocab=3, n_ctx=CTX, act_fn="relu", normalization_type="LN", device=ctx.device, seed=ctx.seed)
    model = HookedTransformer(cfg)
    rand_model = HookedTransformer(cfg)
    rand_model.load_state_dict(model.state_dict())  # control = the initial weights
    eval_seqs = sample(T, 3000, rng)
    eval_B = beliefs(T, eval_seqs)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    steps = 2000 if ctx.quick else 30000
    snaps_at = sorted({0, 100, 300, 1000, 3000, 10000, steps} & set(range(steps + 1)))
    snaps, losses = [], []
    for step in range(steps + 1):
        if step in snaps_at:
            r = regress(model, eval_seqs, eval_B, ctx.device)
            snaps.append({"step": step, "r2": r["r2"], "mse": r["mse"], "xy": simplex_xy(r["pred"])[:: 3]})
            log.info("belief step %d: loss %.4f, belief R² %.4f", step, losses[-1] if losses else float("nan"), r["r2"])
        if step == steps:
            break
        x = torch.tensor(sample(T, 512, rng), device=ctx.device)
        logits = model(x)[:, :-1]
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, 3), x[:, 1:].reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 100 == 0:
            losses.append(loss.item())
    rand = regress(rand_model, eval_seqs, eval_B, ctx.device)
    all_layers = regress(model, eval_seqs, eval_B, ctx.device, layer="all")
    h_opt = entropy_rate(T, rng)
    out = RESULTS / SLUG
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / "mess3_transformer.pt")
    final = snaps[-1]
    summary = {
        "steps": steps,
        "final_r2": final["r2"],
        "final_mse": final["mse"],
        "rand_r2": rand["r2"],
        "all_layers_r2": all_layers["r2"],
        "final_loss": float(np.mean(losses[-10:])),
        "optimal_loss": h_opt,
        "uniform_loss": float(np.log(3)),
    }
    repl = [
        verdict_row(
            "Belief states are linearly decodable from the final residual stream",
            "Mess3: linear regression recovers the belief fractal (MSE ≈ 4e-4, paper Fig. 3)",
            f"held-out R² {final['r2']:.3f} (MSE {final['mse']:.2e}) after {steps} steps vs {rand['r2']:.3f} at initialization",
            "R² ≥ 0.9 and ≥ 0.3 above the random-init control",
            final["r2"] >= 0.9 and final["r2"] - rand["r2"] >= 0.3,
            partial=final["r2"] - rand["r2"] >= 0.2,
        ),
        verdict_row(
            "The model approaches the optimal predictor",
            "trained to near the process's entropy rate",
            f"final loss {summary['final_loss']:.4f} nats vs optimal {h_opt:.4f} and uniform {np.log(3):.4f}",
            "excess loss < 10% of the uniform-to-optimal gap",
            summary["final_loss"] - h_opt < 0.1 * (np.log(3) - h_opt),
        ),
    ]
    payload = {
        "truth_xy": simplex_xy(eval_B.reshape(-1, 3))[::3],
        "truth_rgb": (eval_B.reshape(-1, 3)[::3] * 255).round().astype(int),
        "snaps": snaps,
        "loss_curve": losses,
        "loss_every": 100,
        "rand_xy": simplex_xy(rand["pred"])[::3],
    }
    write_entry(SLUG, payload, summary, repl, models=["4-layer transformer (trained here)"], runtime_s=time.time() - t0, ndigits=3)
    log.info("belief summary: %s", summary)
