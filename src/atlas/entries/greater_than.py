"""Greater-than circuit in GPT-2 small: "The war lasted from the year 1732 to the year 17" -> a year > 32.

Recomputes Hanna, Liu, Variengien 2023, "How does GPT-2 compute greater-than?" (https://arxiv.org/abs/2305.00586):
- probability difference PD = sum_{y > YY} p(y) - sum_{y <= YY} p(y) over the 100 two-digit continuations
  (paper: 81.7% for GPT-2 small);
- the year x next-year probability heatmap has a sharp edge at the start year (paper Fig. 2);
- circuit: heads 5.1, 5.5, 6.9, 7.10, 8.8, 8.11, 9.1 move the start year to the END token, MLPs 8–11 compute
  "years greater than YY" (paper §3–4). We patch components with activations from the corrupted "01" start year
  (the paper's 01-dataset) and measure PD for every on/off combination of {circuit heads, MLP8, MLP9, MLP10, MLP11};
- control: patch random head sets of the same size.
"""

from __future__ import annotations

import itertools
import time

import numpy as np
import torch

from ..common.io import verdict_row, write_entry
from ..common.log import get_logger
from ..common.models import hooked
from ..common.stats import bootstrap_ci, ci_str

log = get_logger()
SLUG = "greater-than"
HEADS = [(5, 1), (5, 5), (6, 9), (7, 10), (8, 8), (8, 11), (9, 1)]
MLPS = [8, 9, 10, 11]
NOUNS = "war deal contract expedition project voyage reign siege trial journey dynasty campaign".split()
CENTURY = "17"


def build(model, quick: bool):
    """Prompts for start years YY in 02..98 whose 4-digit year tokenizes as '17' + 'YY'; corrupted with '01'."""
    years = []
    for yy in range(2, 99):
        toks = model.to_str_tokens(f" {CENTURY}{yy:02d}", prepend_bos=False)
        if len(toks) == 2 and toks[1] == f"{yy:02d}":
            years.append(yy)
    nouns = NOUNS[:3] if quick else NOUNS
    clean, corrupt, yys = [], [], []
    for n in nouns:
        for yy in years:
            clean.append(f"The {n} lasted from the year {CENTURY}{yy:02d} to the year {CENTURY}")
            corrupt.append(f"The {n} lasted from the year {CENTURY}01 to the year {CENTURY}")
            yys.append(yy)
    tc, tx = model.to_tokens(clean), model.to_tokens(corrupt)
    assert tc.shape == tx.shape
    # Token ids for the 100 two-digit continuations "00".."99" (those that are single tokens).
    cont = [model.to_single_token(f"{y:02d}") for y in range(100)]
    return tc, tx, np.array(yys), years, torch.tensor(cont)


def probs_matrix(logits: torch.Tensor, cont: torch.Tensor) -> torch.Tensor:
    """[n, 100] next-token probabilities over the two-digit year tokens (renormalised over those 100)."""
    p = logits[:, -1].softmax(-1)[:, cont.to(logits.device)]
    return p / p.sum(-1, keepdim=True)


def prob_diff(p: torch.Tensor, yys: np.ndarray) -> np.ndarray:
    y = torch.arange(100, device=p.device)[None]
    yy = torch.tensor(yys, device=p.device)[:, None]
    return ((p * (y > yy)).sum(-1) - (p * (y <= yy)).sum(-1)).cpu().numpy()


@torch.no_grad()
def run_patched(model, tc, cache_x, heads, mlps):
    """Forward on clean prompts with the given heads' z and MLP outputs replaced by the corrupted run's."""
    hooks = []
    by_layer: dict[int, list[int]] = {}
    for l, h in heads:
        by_layer.setdefault(l, []).append(h)
    for l, hs in by_layer.items():
        def hz(z, hook, l=l, hs=hs):
            z[:, :, hs] = cache_x["z", l][:, :, hs]
            return z

        hooks.append((f"blocks.{l}.attn.hook_z", hz))
    for l in mlps:
        def hm(x, hook, l=l):
            return cache_x[f"blocks.{l}.hook_mlp_out"]

        hooks.append((f"blocks.{l}.hook_mlp_out", hm))
    return model.run_with_hooks(tc, fwd_hooks=hooks)


@torch.no_grad()
def compute(ctx) -> None:
    t0 = time.time()
    model = hooked("gpt2", device=ctx.device)
    tc_all, tx_all, yys, years, cont = build(model, ctx.quick)
    groups = ["circuit heads"] + [f"MLP {l}" for l in MLPS]
    masks = list(itertools.product([0, 1], repeat=len(groups)))
    yy_index = {yy: i for i, yy in enumerate(years)}
    rng = np.random.default_rng(ctx.seed)
    rand_sets = [[divmod(int(i), 12) for i in rng.choice(144, len(HEADS), replace=False)] for _ in range(10 if ctx.quick else 30)]
    # Faithfulness: patch every attention head NOT in the circuit (all positions); keep all MLPs clean. MLPs 0–7 are not
    # patched because at the YY position they are part of the token's embedding pathway (patching them deletes the
    # start year itself, the circuit's input) — found in the quick smoke test, see reports/experiments/learnings.md.
    others = [(l, h) for l in range(12) for h in range(12) if (l, h) not in HEADS]
    # Accumulators (process prompts in chunks: full logits for ~1,200 prompts would not fit in 8 GB).
    pd_clean, pd_corrupt, pd_circuit_only = [], [], []
    combo_pd = [[] for _ in masks]
    combo_mat = np.zeros((len(masks), len(years), 100))
    cnt = np.zeros(len(years))
    rand_pd = [[] for _ in rand_sets]
    for s in range(0, len(yys), 128):
        tc, tx, yc = tc_all[s : s + 128].to(ctx.device), tx_all[s : s + 128].to(ctx.device), yys[s : s + 128]
        _, cache_x = model.run_with_cache(tx, names_filter=lambda n: n.endswith("hook_z") or n.endswith("hook_mlp_out"))
        p_clean = probs_matrix(model(tc), cont)
        pd_clean.append(prob_diff(p_clean, yc))
        pd_corrupt.append(prob_diff(probs_matrix(model(tx), cont), yc))
        rows = np.array([yy_index[y] for y in yc])
        np.add.at(cnt, rows, 1)
        for mi, mask in enumerate(masks):
            heads = HEADS if mask[0] else []
            mlps = [l for on, l in zip(mask[1:], MLPS) if on]
            p = probs_matrix(run_patched(model, tc, cache_x, heads, mlps), cont) if (heads or mlps) else p_clean
            combo_pd[mi].append(prob_diff(p, yc))
            np.add.at(combo_mat[mi], rows, p.cpu().numpy())
        pd_circuit_only.append(prob_diff(probs_matrix(run_patched(model, tc, cache_x, others, []), cont), yc))
        for ri, rs in enumerate(rand_sets):
            rand_pd[ri].append(prob_diff(probs_matrix(run_patched(model, tc, cache_x, rs, []), cont), yc))
        del cache_x
    pd_clean, pd_corrupt, pd_circuit_only = (np.concatenate(x) for x in (pd_clean, pd_corrupt, pd_circuit_only))
    combos = [{"mask": list(m), "pd": float(np.concatenate(combo_pd[i]).mean()), "matrix": combo_mat[i] / cnt[:, None]} for i, m in enumerate(masks)]
    rand_pd = [float(np.concatenate(r).mean()) for r in rand_pd]

    full = bootstrap_ci(pd_clean)
    circ = bootstrap_ci(pd_circuit_only)
    heads_only = combos[int("10000", 2)]["pd"]
    all_mlps = combos[int("01111", 2)]["pd"]
    everything = combos[-1]["pd"]
    summary = {
        "n_prompts": int(len(yys)),
        "n_years": len(years),
        "pd_clean": full[0],
        "pd_corrupt": float(pd_corrupt.mean()),
        "pd_heads_patched": heads_only,
        "pd_mlps_patched": all_mlps,
        "pd_all_patched": everything,
        "pd_circuit_only": circ[0],
        "faithful_frac": circ[0] / full[0],
        "rand_pd_median": float(np.median(rand_pd)),
    }
    repl = [
        verdict_row(
            "GPT-2 small predicts years greater than the start year",
            "probability difference 81.7%",
            f"PD = {ci_str(*full, pct=True)} over {len(yys)} prompts",
            "PD > 50%",
            full[0] > 0.5,
        ),
        verdict_row(
            "The identified heads + MLPs 8–11 carry the computation",
            "patching the circuit with '01' activations destroys PD; circuit alone keeps 89.5% (72.7 of 81.7 points)",
            # circuit-only here = all 137 non-circuit heads patched, MLPs untouched (see compute())
            f"heads patched → {heads_only * 100:.1f}%, MLPs 8–11 patched → {all_mlps * 100:.1f}%, both → {everything * 100:.1f}%; "
            f"circuit-only → {circ[0] * 100:.1f}% ({circ[0] / full[0] * 100:.0f}% of full)",
            "patching heads+MLPs drops PD below 30% of full, and circuit-only keeps ≥ 60% of full",
            everything < 0.3 * full[0] and circ[0] >= 0.6 * full[0],
            partial=everything < 0.3 * full[0],
        ),
        verdict_row(
            "Random heads are not the circuit",
            "(control, ours)",
            f"patching {len(HEADS)} random heads → median PD {np.median(rand_pd) * 100:.1f}% (circuit heads → {heads_only * 100:.1f}%)",
            "random median PD > circuit-heads PD + 10 points",
            np.median(rand_pd) > heads_only + 0.1,
        ),
    ]
    payload = {
        "years": years,
        "groups": groups,
        "heads": [f"{l}.{h}" for l, h in HEADS],
        "combos": combos,
        "rand_pd": rand_pd,
    }
    write_entry(SLUG, payload, summary, repl, models=["gpt2"], runtime_s=time.time() - t0, ndigits=3)
    log.info("greater-than summary: %s", summary)
