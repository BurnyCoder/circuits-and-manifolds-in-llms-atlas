"""Can SAEs capture concept manifolds? A synthetic benchmark (Bhalla, Fel et al. 2026), recomputed.

Paper: "Do Sparse Autoencoders Capture Concept Manifolds?" (https://arxiv.org/abs/2604.28119, CC BY 4.0; Goodfire post
https://www.goodfire.com/research/can-saes-capture-neural-geometry). Setup (paper App.): several low-dimensional
manifolds (circle, sphere, torus, Möbius strip, Swiss roll, helix, disk, segment), each centered, scaled to unit RMS and
embedded by a random orthonormal map into d = 128; each sample is the sum of L0 = 4 randomly chosen manifolds at random
points, plus tiny noise; a TopK SAE with 512 unit-norm atoms is trained for a sweep of k. They describe three regimes:
capture (a manifold gets about as many atoms as its embedding dimension), shattering (many atoms, each covering a small
patch — local "place cells"), and dilution (many atoms whose responses are spread out/shared).
Our simplified operationalization (stated on the site): support size |G| = atoms firing on ≥ 10% of a manifold's points
when it is shown alone; receptive-field spread = median over those atoms of (mean pairwise distance among the points
an atom fires on) / (mean pairwise distance of the manifold); restricted R² = reconstruction of the manifold from its
best k_i atoms. Regime: capture if |G| ≤ k_i + 2; else shattering if spread < 0.5; else dilution. 2 variants per type
(16 instances) instead of the paper's 6 (48).
"""

from __future__ import annotations

import time

import numpy as np
import torch

from ..common.env import seed_everything
from ..common.io import verdict_row, write_entry
from ..common.log import get_logger

log = get_logger()
SLUG = "sae-manifolds"
D = 128
L0 = 4
ATOMS = 512
KS = [3, 4, 6, 8, 10, 14, 20, 25]


def manifold_points(kind: str, var: int, n: int, rng) -> tuple[np.ndarray, np.ndarray]:
    """Sample n points of one manifold instance -> (embedding coords [n, k], intrinsic coords [n, 1 or 2])."""
    u, v = rng.random(n), rng.random(n)
    t, p = 2 * np.pi * u, 2 * np.pi * v
    r = [0.75, 1.5][var]
    if kind == "circle":
        X = np.c_[r * np.cos(t), r * np.sin(t)]
        I = t[:, None]
    elif kind == "sphere":
        z = 2 * v - 1
        s = np.sqrt(1 - z**2)
        X = r * np.c_[s * np.cos(t), s * np.sin(t), z]
        I = np.c_[t, z]
    elif kind == "torus":
        R, rr = [(2, 0.5), (3, 1)][var]
        X = np.c_[R * np.cos(t), R * np.sin(t), rr * np.cos(p), rr * np.sin(p)]
        I = np.c_[t, p]
    elif kind == "mobius":
        w = [0.3, 0.7][var] * (2 * v - 1)
        X = np.c_[(1 + w * np.cos(t / 2)) * np.cos(t), (1 + w * np.cos(t / 2)) * np.sin(t), w * np.sin(t / 2)]
        I = np.c_[t, w]
    elif kind == "swissroll":
        a = (1.5 + 3 * u) * np.pi * [0.8, 1.0][var]
        X = np.c_[a * np.cos(a), 4 * v, a * np.sin(a)]
        I = np.c_[a, v]
    elif kind == "helix":
        pitch = [0.15, 0.4][var]
        a = 6 * np.pi * u
        X = np.c_[np.cos(a), np.sin(a), pitch * a]
        I = a[:, None]
    elif kind == "disk":
        rad = r * np.sqrt(v)
        X = np.c_[rad * np.cos(t), rad * np.sin(t)]
        I = np.c_[t, rad]
    else:  # segment
        X = (r * (2 * u - 1))[:, None]
        I = X
    return X, I


KINDS = ["circle", "sphere", "torus", "mobius", "swissroll", "helix", "disk", "segment"]


class Instance:
    def __init__(self, kind, var, rng):
        self.kind, self.var = kind, var
        X, _ = manifold_points(kind, var, 50000, rng)
        self.mu = X.mean(0)
        self.scale = np.sqrt(((X - self.mu) ** 2).sum(1).mean())  # unit RMS norm after centering
        self.k = X.shape[1]
        self.V = np.linalg.qr(rng.normal(size=(D, self.k)))[0]  # random orthonormal embedding

    def sample(self, n, rng):
        X, I = manifold_points(self.kind, self.var, n, rng)
        return ((X - self.mu) / self.scale) @ self.V.T, I


class TopKSAE(torch.nn.Module):
    def __init__(self, d, n, k):
        super().__init__()
        self.k = k
        self.enc = torch.nn.Linear(d, n)
        self.dec = torch.nn.Parameter(torch.randn(n, d) / np.sqrt(d))
        self.b_dec = torch.nn.Parameter(torch.zeros(d))

    def codes(self, x):
        z = torch.relu(self.enc(x - self.b_dec))
        top = z.topk(self.k, -1)
        return torch.zeros_like(z).scatter(-1, top.indices, top.values)

    def forward(self, x):
        W = self.dec / self.dec.norm(dim=-1, keepdim=True)  # unit-norm decoder atoms
        return self.codes(x) @ W + self.b_dec


def make_pools(insts, rng, n_pool=20000, device="cpu") -> torch.Tensor:
    """Pre-sampled points of every instance, [n_inst, n_pool, D], so batches are built by indexing (fast)."""
    return torch.tensor(np.stack([i.sample(n_pool, rng)[0] for i in insts]), dtype=torch.float32, device=device)


def train_sae(pools: torch.Tensor, k, steps, device, seed):
    sae = TopKSAE(D, ATOMS, k).to(device)
    opt = torch.optim.Adam(sae.parameters(), lr=3e-3)
    g = torch.Generator(device=device).manual_seed(seed)
    n_inst, n_pool, _ = pools.shape
    for _ in range(steps):
        # Each row: L0 distinct manifolds, each at a random point; tiny Gaussian noise (σ = 1e-5), as in the paper.
        inst_idx = torch.rand(1024, n_inst, generator=g, device=device).argsort(1)[:, :L0]
        pt_idx = torch.randint(0, n_pool, (1024, L0), generator=g, device=device)
        x = pools[inst_idx, pt_idx].sum(1) + 1e-5 * torch.randn(1024, D, generator=g, device=device)
        loss = ((sae(x) - x) ** 2).sum(-1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return sae


@torch.no_grad()
def evaluate(sae, inst, device, rng, n=1500):
    X, I = inst.sample(n, rng)
    x = torch.tensor(X, dtype=torch.float32, device=device)
    z = sae.codes(x).cpu().numpy()
    W = (sae.dec / sae.dec.norm(dim=-1, keepdim=True)).cpu().numpy()
    active = (z > 0).mean(0)
    G = np.where(active >= 0.10)[0]
    # Receptive-field spread of the support atoms.
    sub = rng.choice(n, 400, replace=False)
    Ds = np.linalg.norm(X[sub, None] - X[None, sub], axis=-1)
    base = Ds.mean()
    spreads = []
    for a in G:
        idx = np.where(z[sub, a] > 0)[0]
        if len(idx) >= 2:
            spreads.append(Ds[np.ix_(idx, idx)].mean() / base)
    spread = float(np.median(spreads)) if spreads else 0.0
    # Restricted R²: greedily add the atoms that most reduce residual variance, up to k_i atoms.
    chosen = []
    resid = X.copy()
    for _ in range(inst.k):
        best, best_r = None, None
        for a in G:
            if a in chosen:
                continue
            rec = z[:, chosen + [a]] @ W[chosen + [a]]
            r = ((X - rec) ** 2).sum()
            if best_r is None or r < best_r:
                best, best_r = a, r
        if best is None:
            break
        chosen.append(best)
        resid = X - z[:, chosen] @ W[chosen]
    r2 = 1 - (resid**2).sum() / ((X - X.mean(0)) ** 2).sum()
    regime = "capture" if len(G) <= inst.k + 2 else ("shattering" if spread < 0.5 else "dilution")
    # Viz sample: intrinsic coordinates + which support atom fires most.
    show = sub[:200]
    top_atom = np.where(z[show].max(1) > 0, z[show].argmax(1), -1)
    return {"support": int(len(G)), "spread": spread, "r2": float(r2), "regime": regime, "intr": I[show], "atom": top_atom, "emb3": (X[show] @ inst.V[:, : min(3, inst.k)])}


def compute(ctx) -> None:
    t0 = time.time()
    seed_everything(ctx.seed)
    rng = np.random.default_rng(ctx.seed)
    insts = [Instance(kind, var, rng) for kind in KINDS for var in (0, 1)]
    steps = 300 if ctx.quick else 3000
    ks = KS[:3] if ctx.quick else KS
    table = []
    pools = make_pools(insts, rng, device=ctx.device)
    for k in ks:
        sae = train_sae(pools, k, steps, ctx.device, ctx.seed + k)
        rows = [{"manifold": f"{i.kind}-{i.var}", "kind": i.kind, "k_embed": i.k, **evaluate(sae, i, ctx.device, rng)} for i in insts]
        frac = {r: float(np.mean([x["regime"] == r for x in rows])) for r in ("capture", "shattering", "dilution")}
        table.append({"k": k, "rows": rows, "frac": frac, "r2_mean": float(np.mean([x["r2"] for x in rows]))})
        log.info("SAE k=%d: regimes %s, mean restricted R² %.3f", k, frac, table[-1]["r2_mean"])
    cap = [t["frac"]["capture"] for t in table]
    shat = [t["frac"]["shattering"] for t in table]
    k_best = ks[int(np.argmax(cap))]
    summary = {
        "k_best_capture": k_best,
        "capture_max": float(max(cap)),
        "capture_at_max_k": cap[-1],
        "shatter_at_min_k": shat[0],
        "shatter_at_max_k": shat[-1],
        "k_max": ks[-1],
        "n_instances": len(insts),
        "steps": steps,
    }
    repl = [
        verdict_row(
            "SAEs capture manifolds best when k matches the number of active manifolds",
            "capture peaks near k = 4 = L0 (Bhalla et al. 2026)",
            f"capture fraction peaks at k = {k_best} ({max(cap) * 100:.0f}% of {len(insts)} manifolds)",
            "peak at k ∈ {3, 4, 6}",
            k_best in (3, 4, 6),
        ),
        verdict_row(
            "Larger k shatters manifolds into local atoms",
            "beyond the capture regime, manifolds are tiled by many small receptive fields",
            f"shattering fraction {shat[0] * 100:.0f}% at k = {ks[0]} → {shat[-1] * 100:.0f}% at k = {ks[-1]}; capture {cap[-1] * 100:.0f}% at k = {ks[-1]}",
            "shattering at the largest k ≥ 30 points higher than at the smallest k",
            shat[-1] - shat[0] >= 0.3,
            partial=shat[-1] > shat[0],
        ),
    ]
    write_entry(SLUG, {"ks": ks, "table": table, "kinds": KINDS, "L0": L0, "atoms": ATOMS}, summary, repl, models=["TopK SAEs on synthetic manifolds (trained here)"], runtime_s=time.time() - t0, ndigits=3)
    log.info("sae-manifolds summary: %s", summary)
