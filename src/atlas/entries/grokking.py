"""Grokking modular addition and the Fourier-multiplication circuit (trained from scratch here).

Replicates Nanda, Chan, Lieberum, Smith, Steinhardt 2023, "Progress measures for grokking via mechanistic
interpretability" (https://arxiv.org/abs/2301.05217): a 1-layer ReLU transformer (d_model 128, 4 heads of 32, 512 MLP
neurons, no LayerNorm, untied embed/unembed) trained full-batch with AdamW (lr 1e-3, weight decay 1, betas
(0.9, 0.98)) on 30% of all pairs (a, b) -> (a + b) mod 113 first memorizes, then much later generalizes ("grokking").
The trained network embeds tokens on a few circles (key frequencies k, w_k = 2πk/p) and uses trig identities to
compute cos/sin(w_k(a+b)); the unembedding reads cos(w_k(a+b-c)) (paper §4, Fig. 1).
Hyper-parameters follow the paper's App. / the official repo (github.com/mechanistic-interpretability-grokking/
progress-measures-paper) and ARENA's grokking chapter; 3 seeds + a weight-decay-0 control.
"""

from __future__ import annotations

import time

import numpy as np
import torch

from ..common.env import RESULTS, seed_everything
from ..common.fourier import fourier_basis, freq_norms, project_out_freqs
from ..common.io import verdict_row, write_entry
from ..common.log import get_logger

log = get_logger()
SLUG = "grokking"
P = 113


def make_model(device: str, seed: int):
    from transformer_lens import HookedTransformer, HookedTransformerConfig

    cfg = HookedTransformerConfig(
        n_layers=1, n_heads=4, d_model=128, d_head=32, d_mlp=512, act_fn="relu", normalization_type=None,
        d_vocab=P + 1, d_vocab_out=P, n_ctx=3, init_weights=True, device=device, seed=seed,
    )
    return HookedTransformer(cfg)


def make_data(seed: int, frac_train: float = 0.3):
    """All p^2 pairs as [a, b, '='] (token P is '='); random 30% train split."""
    a, b = torch.meshgrid(torch.arange(P), torch.arange(P), indexing="ij")
    x = torch.stack([a.flatten(), b.flatten(), torch.full((P * P,), P)], 1)
    y = (x[:, 0] + x[:, 1]) % P
    perm = torch.randperm(P * P, generator=torch.Generator().manual_seed(seed))
    n = int(frac_train * P * P)
    return x, y, perm[:n], perm[n:]


def loss_acc(model, x, y):
    logits = model(x)[:, -1].to(torch.float64)  # float64 log-softmax avoids underflow near zero loss (Nanda's code)
    loss = torch.nn.functional.cross_entropy(logits, y)
    return loss, (logits.argmax(-1) == y).float().mean()


def train(device: str, seed: int, epochs: int, wd: float = 1.0, log_every: int = 100, snap_every: int = 500):
    """Full-batch AdamW training; returns (model, curves dict, list of W_E Fourier-norm snapshots)."""
    seed_everything(seed)
    model = make_model(device, seed)
    x, y, tr, te = make_data(seed)
    x, y = x.to(device), y.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=wd, betas=(0.9, 0.98))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(s / 10, 1))  # 10-step warmup as in the paper code
    F, _ = fourier_basis(P, device)
    curves = {k: [] for k in ("epoch", "train_loss", "test_loss", "train_acc", "test_acc", "wnorm")}
    snaps = []
    for ep in range(epochs + 1):
        model.train()
        loss, _ = loss_acc(model, x[tr], y[tr])
        if ep < epochs:
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
        if ep % log_every == 0:
            model.eval()
            with torch.no_grad():
                l_tr, a_tr = loss_acc(model, x[tr], y[tr])
                l_te, a_te = loss_acc(model, x[te], y[te])
                wn = sum((p_.detach() ** 2).sum() for p_ in model.parameters()).item()
            for k, v in zip(curves, (ep, l_tr.item(), l_te.item(), a_tr.item(), a_te.item(), wn)):
                curves[k].append(v)
            if ep % (log_every * 50) == 0:
                log.info("seed %d wd %.1f epoch %5d train %.2e/%.3f test %.2e/%.3f", seed, wd, ep, l_tr, a_tr, l_te, a_te)
        if ep % snap_every == 0:
            with torch.no_grad():
                snaps.append(freq_norms(model.W_E[:P].double(), F).cpu().numpy())
    return model, curves, np.stack(snaps), (x, y, tr, te)


def first_epoch(curves, key, thr):
    for e, v in zip(curves["epoch"], curves[key]):
        if v >= thr:
            return e
    return None


@torch.no_grad()
def analyze(model, data, device):
    """Key frequencies, neuron periodicity, embedding circles, neuron heatmaps and the Fourier ablation."""
    x, y, tr, te = data
    F, _ = fourier_basis(P, device)
    _, cache = model.run_with_cache(x)
    acts = cache["post", 0][:, -1].double()  # [p*p, 512] neuron activations at '='
    grid = acts.reshape(P, P, -1)
    # 2D Fourier transform of each neuron's activation over (a, b): coefficients [p, p, neurons].
    coef = torch.einsum("ia,jb,abn->ijn", F, F, grid)
    tot = (coef[1:, :] ** 2).sum((0, 1)) + (coef[0, 1:] ** 2).sum(0)  # variance (drop the constant term)
    fve = torch.zeros(P // 2, coef.shape[-1], dtype=torch.float64, device=device)
    for k in range(1, P // 2 + 1):
        idx = [0, 2 * k - 1, 2 * k]
        block = coef[idx][:, idx]
        fve[k - 1] = ((block ** 2).sum((0, 1)) - block[0, 0] ** 2) / tot.clamp_min(1e-12)
    best_k = fve.argmax(0) + 1
    best_fve = fve.max(0).values
    periodic = best_fve > 0.85
    ks, counts = torch.unique(best_k[periodic], return_counts=True)
    key = [int(k) for k, c in zip(ks, counts) if c >= 0.02 * periodic.sum()]  # clusters with >= 2% of periodic neurons
    WE = model.W_E[:P].double()
    norms = freq_norms(WE, F)
    frac_top = (norms.sort(descending=True).values ** 2).cumsum(0) / (norms ** 2).sum()
    n90 = int((frac_top < 0.9).sum().item() + 1)

    # Embedding circles: coordinates of each token along the W_E directions of cos/sin of each key frequency.
    circles = {}
    for k in key:
        u = WE.T @ F[2 * k - 1]
        v = WE.T @ F[2 * k]
        circles[str(k)] = torch.stack([WE @ (u / u.norm()), WE @ (v / v.norm())], 1).cpu().numpy()

    # Causal check: remove key frequencies from W_E (test accuracy should collapse) vs an equal number of random others.
    def acc_with(WE_new):
        old = model.W_E.data.clone()
        model.W_E.data[:P] = WE_new.to(old.dtype)
        a = (model(x[te])[:, -1].argmax(-1) == y[te]).float().mean().item()
        model.W_E.data = old
        return a

    rng = np.random.default_rng(0)
    others = [k for k in range(1, P // 2 + 1) if k not in key]
    rand_accs = [acc_with(project_out_freqs(WE, F, list(rng.choice(others, len(key), replace=False)))) for _ in range(10)]
    ablate = {"clean": acc_with(WE), "key removed": acc_with(project_out_freqs(WE, F, key)), "random removed (mean of 10)": float(np.mean(rand_accs))}

    # Heatmaps of 4 neurons from the largest frequency clusters (activation over a, b), scaled to 0..255 ints.
    order = sorted(key, key=lambda k: -int((best_k[periodic] == k).sum()))
    neurons = []
    for k in order[:4]:
        cand = torch.where((best_k == k) & periodic)[0]
        n_ = int(cand[best_fve[cand].argmax()])
        g = grid[:, :, n_]
        g = ((g - g.min()) / (g.max() - g.min() + 1e-12) * 255).round().int().cpu().numpy()
        neurons.append({"neuron": n_, "freq": k, "fve": float(best_fve[n_]), "grid": g})
    return {
        "key_freqs": key,
        "frac_periodic": float(periodic.float().mean()),
        "cluster_sizes": {int(k): int(c) for k, c in zip(ks, counts)},
        "we_norms": norms.cpu().numpy(),
        "n_freq_90": n90,
        "circles": circles,
        "ablate": ablate,
        "neurons": neurons,
    }


def compute(ctx) -> None:
    t0 = time.time()
    epochs = 3000 if ctx.quick else 40000
    seeds = [0, 1, 2] if not ctx.quick else [0]
    runs = []
    for s in seeds:
        model, curves, snaps, data = train(ctx.device, s, epochs)
        an = analyze(model, data, ctx.device)
        out = RESULTS / SLUG
        out.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), out / f"modadd_p113_seed{s}.pt")
        runs.append({"seed": s, "curves": curves, "snaps": snaps, **an})
        log.info("seed %d: key freqs %s, periodic %.2f, n90 %d, ablation %s", s, an["key_freqs"], an["frac_periodic"], an["n_freq_90"], an["ablate"])
    _, wd0_curves, _, _ = train(ctx.device, 0, epochs, wd=0.0)

    tr99 = [first_epoch(r["curves"], "train_acc", 0.99) for r in runs]
    te95 = [first_epoch(r["curves"], "test_acc", 0.95) for r in runs]
    final_te = [r["curves"]["test_acc"][-1] for r in runs]
    delays = [(b / a) if (a and b) else None for a, b in zip(tr99, te95)]
    r0 = runs[0]
    summary = {
        "epochs": epochs,
        "seeds": len(runs),
        "train99_epoch_s0": tr99[0] or -1,
        "test95_epoch_s0": te95[0] or -1,
        "final_test_acc_min": float(min(final_te)),
        "key_freqs_s0": ", ".join(map(str, r0["key_freqs"])),
        "key_freqs_all": " | ".join(", ".join(map(str, r["key_freqs"])) for r in runs),
        "frac_periodic_s0": r0["frac_periodic"],
        "n_freq_90_max": max(r["n_freq_90"] for r in runs),
        "abl_clean": r0["ablate"]["clean"],
        "abl_key": r0["ablate"]["key removed"],
        "abl_rand": r0["ablate"]["random removed (mean of 10)"],
        "wd0_final_test_acc": wd0_curves["test_acc"][-1],
        "wd0_final_train_acc": wd0_curves["train_acc"][-1],
    }
    grok_ok = all(d is not None and d >= 3 for d in delays) and min(final_te) > 0.99
    repl = [
        verdict_row(
            "Generalization comes long after memorization (grokking)",
            "train acc ~100% by ~1.4k epochs; test generalizes at ~10k (Fig. 2, wd=1, 30% split)",
            "; ".join(f"seed {r['seed']}: train≥99% @ {a}, test≥95% @ {b}" for r, a, b in zip(runs, tr99, te95)),
            "test-95% epoch ≥ 3× train-99% epoch and final test acc > 99%, for every seed",
            grok_ok,
        ),
        verdict_row(
            "The embedding is sparse in the Fourier basis",
            "W_E has ~6 non-negligible frequencies of 56 (§5.1)",
            f"≤ {summary['n_freq_90_max']} frequencies carry 90% of ‖W_E‖² (worst seed)",
            "≤ 10 frequencies carry 90% of the norm in every seed",
            summary["n_freq_90_max"] <= 10,
        ),
        verdict_row(
            "A handful of key frequencies; most neurons are single-frequency",
            "5 key freqs {14, 35, 41, 42, 52} in their seed; 84.6% of neurons > 85% FVE from one frequency",
            f"key freqs per seed: {summary['key_freqs_all']}; {r0['frac_periodic'] * 100:.1f}% of neurons periodic (seed 0)",
            "3–7 key frequencies per seed and > 60% periodic neurons (exact frequencies are seed-dependent)",
            all(3 <= len(r["key_freqs"]) <= 7 for r in runs) and r0["frac_periodic"] > 0.6,
        ),
        verdict_row(
            "The key frequencies are causally necessary",
            "ablating key frequencies destroys performance; other frequencies don't matter (§5.2, Fig. 6)",
            f"test acc {summary['abl_clean']:.3f} → {summary['abl_key']:.3f} (key removed) vs {summary['abl_rand']:.3f} (random removed)",
            "key removal → < 20% acc; random removal → > 90%",
            summary["abl_key"] < 0.2 and summary["abl_rand"] > 0.9,
        ),
        verdict_row(
            "Weight decay is needed for grokking",
            "no grokking without regularization in this setup (§5.3 / App.)",
            f"wd=0: final train {summary['wd0_final_train_acc']:.3f}, test {summary['wd0_final_test_acc']:.3f} after {epochs} epochs",
            "wd=0 test accuracy stays < 50%",
            summary["wd0_final_test_acc"] < 0.5,
        ),
    ]
    payload = {
        "runs": [
            {k: r[k] for k in ("seed", "curves", "key_freqs", "frac_periodic", "cluster_sizes", "we_norms", "n_freq_90", "ablate")}
            for r in runs
        ],
        "wd0": wd0_curves,
        "snaps": r0["snaps"],  # [n_snapshots, 56] W_E Fourier norms over training (seed 0)
        "snap_every": 500,
        "circles": r0["circles"],
        "neurons": r0["neurons"],
    }
    write_entry(SLUG, payload, summary, repl, models=["1L transformer (trained here)"], runtime_s=time.time() - t0, ndigits=4)
    log.info("grokking summary: %s", summary)
