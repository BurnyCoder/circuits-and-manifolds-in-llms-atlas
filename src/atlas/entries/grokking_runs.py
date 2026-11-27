"""Cross-seed grokking statistics from the companion project "Grokking, Visualized" (22 runs, MIT license).

Source: https://github.com/BurnyCoder/visualizing-grokking-modular-addition-in-transformer — models and per-run analysis
on Hugging Face (``BurnyCoder/grokking-modular-addition-runs``): 11 seeds with weight decay 1 (``wd1_s*``), 5 without
weight decay (``wd0_s*``), 3 random-label controls (``rand_s*``) and 3 runs with the paper's split but different inits
(``split0_init*``), all trained with Nanda et al.'s (2023) settings. We download only the small per-run files
(``meta.json``, ``curves.npz``, ``analysis/{phases,final}.json``; formats in that repo's ``docs/data-formats.md``), not
the 360 MB checkpoints, and summarize them for the grokking entry: grokking step per seed, key frequencies,
single-frequency-neuron fraction, logit Fourier fit, and the controls' outcomes. These are the companion project's
measurements, re-aggregated here; our own 3-seed retraining lives in ``grokking.py``.
"""

from __future__ import annotations

import json
import time

import numpy as np

from ..common.io import verdict_row, write_entry
from ..common.log import get_logger

log = get_logger()
SLUG = "grokking-runs"
REPO = "BurnyCoder/grokking-modular-addition-runs"
RUNS = (
    [f"wd1_s{i}" for i in range(11)]
    + [f"wd0_s{i}" for i in range(5)]
    + [f"rand_s{i}" for i in range(3)]
    + [f"split0_init{i}" for i in (100, 101, 102)]
)


def _get(run: str, name: str) -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(REPO, f"{run}/{name}")


def compute(ctx) -> None:
    t0 = time.time()
    runs = []
    for r in RUNS[:4] if ctx.quick else RUNS:
        meta = json.load(open(_get(r, "meta.json"), encoding="utf-8"))
        phases = json.load(open(_get(r, "analysis/phases.json"), encoding="utf-8"))
        final = json.load(open(_get(r, "analysis/final.json"), encoding="utf-8"))
        z = np.load(_get(r, "curves.npz"))
        te_steps, te_acc = z["test_loss_steps"], z["test_acc"]
        tr_acc, tr_steps = z["train_acc"], z["train_loss_steps"]
        idx = np.linspace(0, len(te_steps) - 1, 200).astype(int)  # downsample curves to 200 points for the site
        tri = np.searchsorted(tr_steps, te_steps[idx])
        first = lambda arr, steps, thr: int(steps[np.argmax(arr >= thr)]) if (arr >= thr).any() else None  # noqa: E731
        runs.append(
            {
                "id": r,
                "group": meta["group"],
                "weight_decay": meta["weight_decay"],
                "split_seed": meta["split_seed"],
                "init_seed": meta["init_seed"],
                # The companion pipeline lists every frequency as "key" when the spectrum has no sparse structure
                # (wd = 0 and random-label runs); store that as an empty list + a 'diffuse' flag.
                "key_freqs": final.get("key_freqs", []) if len(final.get("key_freqs", [])) <= 10 else [],
                "diffuse": len(final.get("key_freqs", [])) > 10,
                "frac_neurons_085": final.get("frac_neurons_085"),
                "logit_fve": final.get("logit_fve"),
                "gini_we": final.get("gini_we"),
                "t_grok": phases.get("t_grok"),
                "train99": first(tr_acc, tr_steps, 0.99),
                "test95": first(te_acc, te_steps, 0.95),
                "final_test_acc": float(te_acc[-1]),
                "curve": {"step": te_steps[idx], "test_acc": te_acc[idx], "train_acc": tr_acc[np.clip(tri, 0, len(tr_acc) - 1)]},
            }
        )
        log.info("%s: key freqs %s, grok@%s, final test %.3f", r, runs[-1]["key_freqs"], runs[-1]["test95"], runs[-1]["final_test_acc"])

    wd1 = [r for r in runs if r["group"] == "wd1"]
    wd0 = [r for r in runs if r["group"] == "wd0"]
    rnd = [r for r in runs if r["group"] == "rand"]
    grokked = [r for r in wd1 if r["test95"] and r["final_test_acc"] > 0.99]
    delays = [r["test95"] / r["train99"] for r in grokked if r["train99"]]
    nkey = [len(r["key_freqs"]) for r in wd1]
    fracn = [r["frac_neurons_085"] for r in wd1 if r["frac_neurons_085"] is not None]
    summary = {
        "n_runs": len(runs),
        "n_wd1": len(wd1),
        "n_wd1_grokked": len(grokked),
        "delay_median": float(np.median(delays)) if delays else -1,
        "delay_min": float(np.min(delays)) if delays else -1,
        "nkey_min": int(min(nkey)) if nkey else -1,
        "nkey_max": int(max(nkey)) if nkey else -1,
        "frac_neurons_median": float(np.median(fracn)) if fracn else -1,
        "n_wd0": len(wd0),
        "wd0_max_test": float(max((r["final_test_acc"] for r in wd0), default=-1)),
        "n_rand": len(rnd),
        "rand_max_test": float(max((r["final_test_acc"] for r in rnd), default=-1)),
    }
    repl = [
        verdict_row(
            "Grokking replicates across seeds (companion project's 11 runs)",
            "grokking with wd = 1 on a 30% split (Nanda et al. 2023, one run)",
            f"{len(grokked)}/{len(wd1)} seeds reach > 99% test accuracy; test-95% step ÷ train-99% step: median {summary['delay_median']:.1f}× (min {summary['delay_min']:.1f}×)",
            "≥ 9 of 11 seeds grok, with delay ≥ 3× in each",
            len(grokked) >= 9 and (min(delays) if delays else 0) >= 3,
        ),
        verdict_row(
            "A handful of key frequencies and single-frequency neurons in every seed",
            "5 key frequencies; 84.6% of neurons single-frequency",
            f"key frequencies per seed: {summary['nkey_min']}–{summary['nkey_max']}; median {summary['frac_neurons_median'] * 100:.1f}% of neurons > 85% FVE",
            "3–7 key frequencies in every seed and median single-frequency fraction > 60%",
            summary["nkey_min"] >= 3 and summary["nkey_max"] <= 7 and summary["frac_neurons_median"] > 0.6,
        ),
        verdict_row(
            "Controls: no weight decay / random labels don't generalize",
            "no grokking without regularization; nothing to learn with random labels",
            f"wd = 0: best final test acc {summary['wd0_max_test'] * 100:.1f}% over {len(wd0)} runs; random labels: {summary['rand_max_test'] * 100:.1f}% over {len(rnd)}",
            "every wd = 0 run < 50% and every random-label run < 5% test accuracy",
            summary["wd0_max_test"] < 0.5 and summary["rand_max_test"] < 0.05,
        ),
    ]
    write_entry(SLUG, {"runs": runs, "source": f"https://huggingface.co/{REPO}"}, summary, repl, models=[REPO], runtime_s=time.time() - t0)
    log.info("grokking runs summary: %s", summary)
