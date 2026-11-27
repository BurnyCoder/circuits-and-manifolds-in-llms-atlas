"""Can SAEs capture concept manifolds? A synthetic benchmark (Bhalla, Fel et al. 2026), recomputed.

Paper: "Do Sparse Autoencoders Capture Concept Manifolds?" (https://arxiv.org/abs/2604.28119, CC BY 4.0; Goodfire post
https://www.goodfire.com/research/can-saes-capture-neural-geometry). Setup (paper App.): several low-dimensional
manifolds (circle, sphere, torus, Möbius strip, Swiss roll, helix, disk, segment), each centered, scaled to unit RMS and
embedded by a random orthonormal map into d = 128; each sample is the sum of L0 = 4 randomly chosen manifolds at random
points, plus tiny noise; a TopK SAE with 512 unit-norm atoms is trained for a sweep of k. They describe three regimes:
capture (a manifold gets about as many atoms as its embedding dimension), shattering (many atoms, each covering a small
patch — local "place cells"), and dilution (many atoms whose responses are spread out/shared).
Metrics follow the paper's App. E, on in-distribution (superposed) evaluation inputs: for manifold i, the rows where it
is active and its true contribution M_i there.
- Restricted R² (the paper's subspace-capture score): greedily pick k_i decoder directions that explain the most residual
  variance of M_i, keep only those atoms' codes, decode, and compute R² against M_i.
- Support size |S|: atoms firing on ≥ 10% of the manifold's rows (after discarding each atom's activations below the
  10th percentile of its nonzero activations) and on ≥ 30 rows.
- Receptive-field spread: median over support atoms of the mean pairwise distance between the M_i points where the atom
  fires, normalized by the manifold's own mean pairwise distance.
Regime label (ours; the paper shows a phase diagram of support size × spread): capture if |S| ≤ k_i + 2, else shattering
if spread < 0.5, else dilution. 8 types × 6 variants = 48 instances, as in the paper (with 16 instances any other
manifold co-occurs in ~20% of rows, so the 10% support threshold would count its atoms too).
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


VARIANTS = 6  # per manifold type, as in the paper (48 instances in total)


def manifold_points(kind: str, var: int, n: int, rng) -> tuple[np.ndarray, np.ndarray]:
    """Sample n points of one manifold instance -> (embedding coords [n, k], intrinsic coords [n, 1 or 2]).
    `var` ∈ 0..VARIANTS−1 interpolates each type's shape parameters (radius, tube ratio, width, pitch)."""
    u, v = rng.random(n), rng.random(n)
    t, p = 2 * np.pi * u, 2 * np.pi * v
    f = var / (VARIANTS - 1)
    r = 0.75 + 0.75 * f
    if kind == "circle":
        X = np.c_[r * np.cos(t), r * np.sin(t)]
        I = t[:, None]
    elif kind == "sphere":
        z = 2 * v - 1
        s = np.sqrt(1 - z**2)
        X = r * np.c_[s * np.cos(t), s * np.sin(t), z]
        I = np.c_[t, z]
    elif kind == "torus":
        R, rr = 2 + f, 0.5 + 0.5 * f
        X = np.c_[R * np.cos(t), R * np.sin(t), rr * np.cos(p), rr * np.sin(p)]
        I = np.c_[t, p]
    elif kind == "mobius":
        w = (0.3 + 0.4 * f) * (2 * v - 1)
        X = np.c_[(1 + w * np.cos(t / 2)) * np.cos(t), (1 + w * np.cos(t / 2)) * np.sin(t), w * np.sin(t / 2)]
        I = np.c_[t, w]
    elif kind == "swissroll":
        a = (1.5 + 3 * u) * np.pi * (0.8 + 0.2 * f)
        X = np.c_[a * np.cos(a), 4 * v, a * np.sin(a)]
        I = np.c_[a, v]
    elif kind == "helix":
        pitch = 0.15 + 0.25 * f
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


def make_pools(insts, rng, n_pool=20000, device="cpu") -> tuple[torch.Tensor, np.ndarray]:
    """Pre-sampled points of every instance: embeddings [n_inst, n_pool, D] (GPU; batches are built by indexing) and
    intrinsic coordinates [n_inst, n_pool, 2] (CPU; 1-D manifolds padded with zeros)."""
    X, I = zip(*[i.sample(n_pool, rng) for i in insts])
    intr = np.stack([np.c_[c, np.zeros((len(c), 2 - c.shape[1]))] if c.shape[1] < 2 else c for c in I])
    return torch.tensor(np.stack(X), dtype=torch.float32, device=device), intr


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
def evaluate_all(sae, insts, pools, intr, device, seed, n_eval=24000) -> list[dict]:
    """Paper's App. E metrics for every instance, on superposed evaluation inputs (see module docstring)."""
    g = torch.Generator(device=device).manual_seed(seed)
    n_inst, n_pool, _ = pools.shape
    inst_idx = torch.rand(n_eval, n_inst, generator=g, device=device).argsort(1)[:, :L0]  # [n, L0] active manifolds
    pt_idx = torch.randint(0, n_pool, (n_eval, L0), generator=g, device=device)
    contrib = pools[inst_idx, pt_idx]  # [n, L0, D] true contributions
    x = contrib.sum(1) + 1e-5 * torch.randn(n_eval, D, generator=g, device=device)
    Z = torch.cat([sae.codes(x[i : i + 4096]) for i in range(0, n_eval, 4096)])  # [n, atoms]
    W = sae.dec / sae.dec.norm(dim=-1, keepdim=True)  # unit decoder atoms [atoms, D]
    # Per-atom magnitude floor: 10th percentile of each atom's nonzero activations (paper App. E).
    Zc = Z.cpu().numpy()
    thr = np.array([np.quantile(c[c > 0], 0.10) if (c > 0).sum() >= 10 else np.inf for c in Zc.T])
    on = Zc >= thr[None]  # [n, atoms] robust firing
    inst_np, pt_np = inst_idx.cpu().numpy(), pt_idx.cpu().numpy()
    rng = np.random.default_rng(seed)
    rows_out = []
    for i, inst in enumerate(insts):
        rows, slot = np.where(inst_np == i)
        M = contrib[torch.tensor(rows, device=device), torch.tensor(slot, device=device)]  # [n_i, D]
        Zi = Z[torch.tensor(rows, device=device)]
        # Restricted R² at n = k_i: greedy decoder directions explaining the residual variance of M_i.
        resid, chosen = M.clone(), []
        for _ in range(inst.k):
            gain = ((resid @ W.T) ** 2).sum(0)
            if chosen:
                gain[chosen] = -1
            j = int(gain.argmax())
            chosen.append(j)
            resid = resid - (resid @ W[j])[:, None] * W[j][None]
        M_hat = Zi[:, chosen] @ W[chosen]
        r2 = float(1 - ((M - M_hat) ** 2).sum() / ((M - M.mean(0)) ** 2).sum())
        # Support size and receptive-field spread on the manifold's rows.
        on_i = on[rows]
        S = np.where((on_i.mean(0) >= 0.10) & (on_i.sum(0) >= 30))[0]
        Mn = M.cpu().numpy()
        sub = rng.choice(len(Mn), min(600, len(Mn)), replace=False)
        base = np.linalg.norm(Mn[sub, None] - Mn[None, sub], axis=-1).mean()
        spreads = []
        for a in S:
            idx = np.where(on_i[:, a])[0]
            idx = rng.choice(idx, min(300, len(idx)), replace=False)
            if len(idx) >= 2:
                spreads.append(np.linalg.norm(Mn[idx, None] - Mn[None, idx], axis=-1).mean() / base)
        spread = float(np.median(spreads)) if spreads else 0.0
        regime = "capture" if len(S) <= inst.k + 2 else ("shattering" if spread < 0.5 else "dilution")
        # Viz: 200 of the manifold's points, colored by their most active support atom (−1 = none).
        show = sub[:200]
        top = np.full(len(show), -1)
        if len(S):
            zs = Zc[rows[show]][:, S]
            top = np.where(zs.max(1) > 0, S[zs.argmax(1)], -1)
        rows_out.append({
            "manifold": f"{inst.kind}-{inst.var}", "kind": inst.kind, "k_embed": inst.k, "support": int(len(S)), "spread": spread,
            "r2": r2, "regime": regime, "intr": intr[i, pt_np[rows[show], slot[show]]],
            "atom": top, "emb3": (Mn[show] @ inst.V[:, : min(3, inst.k)]),
        })
    return rows_out


def compute(ctx) -> None:
    t0 = time.time()
    seed_everything(ctx.seed)
    rng = np.random.default_rng(ctx.seed)
    insts = [Instance(kind, var, rng) for kind in KINDS for var in range(VARIANTS)]
    steps = 300 if ctx.quick else 3000
    ks = KS[:3] if ctx.quick else KS
    table = []
    pools, intr = make_pools(insts, rng, device=ctx.device)
    for k in ks:
        sae = train_sae(pools, k, steps, ctx.device, ctx.seed + k)
        rows = evaluate_all(sae, insts, pools, intr, ctx.device, ctx.seed + 1000 + k)
        frac = {r: float(np.mean([x["regime"] == r for x in rows])) for r in ("capture", "shattering", "dilution")}
        table.append({"k": k, "rows": rows, "frac": frac, "r2_mean": float(np.mean([x["r2"] for x in rows]))})
        log.info("SAE k=%d: regimes %s, mean restricted R² %.3f", k, frac, table[-1]["r2_mean"])
    cap = [t["frac"]["capture"] for t in table]
    shat = [t["frac"]["shattering"] for t in table]
    dil = [t["frac"]["dilution"] for t in table]
    r2s = [t["r2_mean"] for t in table]
    k_best = ks[int(np.argmax(r2s))]
    summary = {
        "k_best_r2": k_best,
        "r2_best": float(max(r2s)),
        "r2_min_k": r2s[0],
        "r2_max_k": r2s[-1],
        "capture_max": float(max(cap)),
        "k_best_capture": ks[int(np.argmax(cap))] if max(cap) > 0 else -1,
        "shatter_at_min_k": shat[0],
        "shatter_at_max_k": shat[-1],
        "dilution_at_min_k": dil[0],
        "dilution_at_max_k": dil[-1],
        "k_max": ks[-1],
        "n_instances": len(insts),
        "steps": steps,
    }
    repl = [
        verdict_row(
            "Subspace capture has a sparsity sweet spot",
            "restricted R² (k_i atoms reconstruct each manifold from superposed codes) peaks near k = 4 (Bhalla et al. 2026, Fig. 4A)",
            f"mean restricted R² peaks at k = {k_best} ({max(r2s):.2f}; {r2s[0]:.2f} at k = {ks[0]}, {r2s[-1]:.2f} at k = {ks[-1]}); "
            f"regime label 'capture' for up to {max(cap) * 100:.0f}% of {len(insts)} manifolds",
            "restricted-R² peak at k ∈ {3, 4, 6} (amendment 2026-11-27: the paper's capture score; was 'capture fraction')",
            k_best in (3, 4, 6),
            partial=k_best == 8,
        ),
        verdict_row(
            "Sparsity moves the SAE from shattering (small k) to dilution (large k)",
            "low k: shattered across atoms; intermediate k: capture; high k: redundant atoms lose specificity (dilution) (Fig. 4B)",
            f"shattering {shat[0] * 100:.0f}% at k = {ks[0]} → {shat[-1] * 100:.0f}% at k = {ks[-1]}; dilution {dil[0] * 100:.0f}% → {dil[-1] * 100:.0f}%",
            "dilution at the largest k ≥ 30 points above the smallest k and shattering higher at the smallest k (amendment: the "
            "pre-stated claim had the direction reversed relative to the paper)",
            dil[-1] - dil[0] >= 0.3 and shat[0] > shat[-1],
            partial=dil[-1] > dil[0] and shat[0] > shat[-1],
        ),
    ]
    write_entry(SLUG, {"ks": ks, "table": table, "kinds": KINDS, "L0": L0, "atoms": ATOMS}, summary, repl, models=["TopK SAEs on synthetic manifolds (trained here)"], runtime_s=time.time() - t0, ndigits=3)
    log.info("sae-manifolds summary: %s", summary)
