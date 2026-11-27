"""Block-sparse featurizers: concepts are 2–4-dimensional blocks, not single directions (Fel et al. 2026), recomputed.

Paper: "Structuring Sparsity: Block-Sparse Featurizers Capture Visual Concept Manifolds" (https://arxiv.org/abs/2606.25234,
CC BY 4.0; Goodfire post https://www.goodfire.com/research/bsf-vision; MIT code github.com/goodfire-ai/block-sparse-featurizer).
A block-sparse featurizer (BSF) groups latents into blocks of size b and keeps the top-k blocks by L2 norm:
z = Π_k(xW + b), x̂ = Σ_g z_g D_g — so a concept can be a small subspace (position *within* the concept = direction
inside the block). No ReLU inside blocks (it would restrict a block to a cone).
1) Toy (paper's setup, scaled down): M concept manifolds, each points on a sphere in its own random 3-D subspace of
   d = 128; each sample sums L0 = 4 of them. Compare a vanilla BSF (G = 2M blocks of b = 4, top-4 blocks) to a TopK SAE
   with the same number of latents and active latents (k = 16): per-concept R² of reconstructing that concept's
   component from its best block (BSF) or its best 4 atoms (SAE). Paper: BSF 0.93–0.97 vs TopK SAE ≈ 0.53.
2) Curve "block" in InceptionV1: the paper finds one b = 16 block covering all curve orientations whose activations are
   89% explained by the first three Fourier harmonics of orientation. Our stand-in for the learned block: the top-16
   principal components of mixed3b's response to curve stimuli at 72 orientations; we report the variance explained
   by orientation harmonics 1–3 (vs a shuffled-orientation null).
"""

from __future__ import annotations

import time

import numpy as np
import torch

from ..common.env import seed_everything
from ..common.io import verdict_row, write_entry
from ..common.log import get_logger

log = get_logger()
SLUG = "block-sparse-featurizers"
D, M, L0, B, K_BLOCKS = 128, 32, 4, 4, 4


class BSF(torch.nn.Module):
    def __init__(self, d, groups, b, k):
        super().__init__()
        self.g, self.b, self.k = groups, b, k
        self.enc = torch.nn.Linear(d, groups * b)
        self.dec = torch.nn.Parameter(torch.randn(groups * b, d) / np.sqrt(d))
        self.b_dec = torch.nn.Parameter(torch.zeros(d))

    def codes(self, x):
        z = self.enc(x - self.b_dec).view(-1, self.g, self.b)
        keep = z.norm(dim=-1).topk(self.k, -1).indices
        mask = torch.zeros(z.shape[:2], device=x.device).scatter(-1, keep, 1.0)
        return z * mask[..., None]

    def forward(self, x):
        return self.codes(x).flatten(1) @ self.dec + self.b_dec


class TopK(torch.nn.Module):
    def __init__(self, d, n, k):
        super().__init__()
        self.k = k
        self.enc = torch.nn.Linear(d, n)
        self.dec = torch.nn.Parameter(torch.randn(n, d) / np.sqrt(d))
        self.b_dec = torch.nn.Parameter(torch.zeros(d))

    def codes(self, x):
        z = torch.relu(self.enc(x - self.b_dec))
        t = z.topk(self.k, -1)
        return torch.zeros_like(z).scatter(-1, t.indices, t.values)

    def forward(self, x):
        return self.codes(x) @ self.dec + self.b_dec


def toy(ctx) -> dict:
    g = torch.Generator(device=ctx.device).manual_seed(ctx.seed)
    V = torch.linalg.qr(torch.randn(M, D, 3, generator=g, device=ctx.device))[0]  # [M, D, 3] orthonormal subspaces

    def batch(n):
        idx = torch.rand(n, M, generator=g, device=ctx.device).argsort(1)[:, :L0]
        s = torch.randn(n, L0, 3, generator=g, device=ctx.device)
        s = s / s.norm(dim=-1, keepdim=True)  # points on the unit sphere of each concept
        comp = torch.einsum("nlk,nldk->nld", s, V[idx])  # [n, L0, D]
        return comp.sum(1), idx, comp

    steps = 400 if ctx.quick else 6000
    models = {"BSF (blocks of 4)": BSF(D, 2 * M, B, K_BLOCKS).to(ctx.device), "TopK SAE (k = 16)": TopK(D, 2 * M * B, K_BLOCKS * B).to(ctx.device)}
    for name, mdl in models.items():
        opt = torch.optim.Adam(mdl.parameters(), lr=1e-3)
        for _ in range(steps):
            x, _, _ = batch(1024)
            loss = ((mdl(x) - x) ** 2).sum(-1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        log.info("%s trained, final loss %.4f", name, loss.item())
    # Per-concept R²: show each concept alone; best block (BSF) / best 4 atoms (SAE) reconstruct its component.
    res = {name: [] for name in models}
    with torch.no_grad():
        for c in range(M):
            s = torch.randn(2000, 3, generator=g, device=ctx.device)
            s = s / s.norm(dim=-1, keepdim=True)
            x = s @ V[c].T
            for name, mdl in models.items():
                z = mdl.codes(x)
                if isinstance(mdl, BSF):
                    blk = z.norm(dim=-1).mean(0).argmax()
                    rec = z[:, blk] @ mdl.dec.view(mdl.g, mdl.b, D)[blk] + mdl.b_dec
                else:
                    top = z.mean(0).topk(B).indices
                    rec = z[:, top] @ mdl.dec[top] + mdl.b_dec
                r2 = 1 - ((x - rec) ** 2).sum() / ((x - x.mean(0)) ** 2).sum()
                res[name].append(float(r2))
    return {k: v for k, v in res.items()}


@torch.no_grad()
def curve_block(ctx) -> dict:
    from lucent.modelzoo import inceptionv1

    from .curve_detectors import ORIENTS, RADII, arc_image

    model = inceptionv1(pretrained=True).to(ctx.device).eval()
    store = {}
    h = model.mixed3b.register_forward_hook(lambda m, i, o: store.__setitem__("a", o))
    feats, ors = [], []
    for r in RADII:
        imgs = np.stack([arc_image(float(t), r) for t in ORIENTS])
        model(torch.tensor(imgs, device=ctx.device) * 255 - 117)
        a = store["a"]
        c = a.shape[-1] // 2
        feats.append(a[:, :, c - 1 : c + 2, c - 1 : c + 2].mean((-1, -2)).cpu().numpy())
        ors.append(ORIENTS)
    h.remove()
    X = np.concatenate(feats)  # [n_stim, 480]
    th = np.radians(np.concatenate(ors))
    Xc = X - X.mean(0)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    Zb = Xc @ Vt[:16].T  # 16-D "block" stand-in

    def harm_r2(theta):
        Hb = np.c_[[f(m * theta) for m in (1, 2, 3) for f in (np.cos, np.sin)]].T
        Hb = np.c_[Hb, np.ones(len(theta))]
        W, *_ = np.linalg.lstsq(Hb, Zb, rcond=None)
        out = []
        for m in range(3):
            cols = [2 * m, 2 * m + 1]
            out.append(float(((Hb[:, cols] @ W[cols]) ** 2).sum() / (Zb**2).sum()))
        return out

    per = harm_r2(th)
    rng = np.random.default_rng(0)
    null = [sum(harm_r2(rng.permutation(th))) for _ in range(50)]
    return {"harmonics": per, "total": float(sum(per)), "null_mean": float(np.mean(null)), "null_p95": float(np.quantile(null, 0.95)), "ring": Zb[:, :2], "orient_deg": np.degrees(th)}


def compute(ctx) -> None:
    t0 = time.time()
    seed_everything(ctx.seed)
    t = toy(ctx)
    cb = curve_block(ctx)
    names = list(t)
    bsf, sae = np.array(t[names[0]]), np.array(t[names[1]])
    summary = {
        "bsf_r2_median": float(np.median(bsf)),
        "sae_r2_median": float(np.median(sae)),
        "n_concepts": M,
        "curve_h1": cb["harmonics"][0],
        "curve_h123": cb["total"],
        "curve_null_p95": cb["null_p95"],
    }
    repl = [
        verdict_row(
            "Blocks capture multi-dimensional concepts that single-direction SAEs fragment",
            "per-block R²: BSF 0.93–0.97 vs TopK SAE ≈ 0.53 (toy, Fel et al. 2026)",
            f"median per-concept R²: BSF {np.median(bsf):.2f} vs TopK SAE {np.median(sae):.2f} ({M} sphere concepts in 3-D subspaces of d = 128)",
            "BSF median ≥ 0.8 and ≥ 0.2 above the SAE",
            np.median(bsf) >= 0.8 and np.median(bsf) - np.median(sae) >= 0.2,
            partial=np.median(bsf) > np.median(sae),
        ),
        verdict_row(
            "InceptionV1 curve responses form an orientation ring described by a few harmonics",
            "one b = 16 block covers all orientations; harmonics 1–3 explain 59/18/12% (89%)",
            f"top-16 PCs of mixed3b curve responses: harmonics 1–3 explain {cb['harmonics'][0] * 100:.0f}/{cb['harmonics'][1] * 100:.0f}/{cb['harmonics'][2] * 100:.0f}% "
            f"({cb['total'] * 100:.0f}%; shuffled-orientation null 95th pct {cb['null_p95'] * 100:.0f}%)",
            "harmonics 1–3 ≥ 50% and above the null's 95th percentile",
            cb["total"] >= 0.5 and cb["total"] > cb["null_p95"],
            partial=cb["total"] > cb["null_p95"],
        ),
    ]
    write_entry(SLUG, {"toy": t, "curve": cb}, summary, repl, models=["BSF / TopK SAE (trained here)", "InceptionV1"], runtime_s=time.time() - t0, ndigits=3)
    log.info("BSF summary: %s", summary)
