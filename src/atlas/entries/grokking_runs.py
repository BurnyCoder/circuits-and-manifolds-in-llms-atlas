"""Cross-seed grokking statistics from the companion project "Grokking, Visualized" (22 runs, MIT license).

Source: https://github.com/BurnyCoder/visualizing-grokking-modular-addition-in-transformer — models and per-run analysis
on Hugging Face (``BurnyCoder/grokking-modular-addition-runs``): 11 seeds with weight decay 1 (``wd1_s*``), 5 without
weight decay (``wd0_s*``), 3 random-label controls (``rand_s*``) and 3 runs with the paper's split but different inits
(``split0_init*``), all trained with Nanda et al.'s (2023) settings. We download only the small per-run files
(``meta.json``, ``curves.npz``, ``analysis/{phases,final}.json``; formats in that repo's ``docs/data-formats.md``), not
the 360 MB checkpoints, and summarize them for the grokking entry: grokking step per seed, key frequencies,
single-frequency-neuron fraction, logit Fourier fit, and the controls' outcomes. These are the companion project's
measurements, re-aggregated here; our own 3-seed retraining lives in ``grokking.py``.

Progress measures (Nanda et al. 2023, §5): for the 11 weight-decay seeds we also download the per-checkpoint metrics
(``analysis/ckpt_metrics.npz``, 401 checkpoints every 100 steps, ~12 MB per run) and re-implement the companion's
preregistered H1/H2 tests from its ``docs/operationalizations.md``:
- H1: the restricted loss (logits projected onto the key frequencies, all pairs) reaches 10% of its value at t1 before
  the test loss does (a run whose test loss never drops tenfold passes if the restricted loss does);
- H2: the excluded loss (train pairs, key frequencies removed) at t2 is > 10x its value at t1;
with t1 (end of memorization) and t2 (start of generalization) from ``analysis/phases.json``; quantities "at t" are read
at the first checkpoint at or after t. The companion's own verdicts (H1-H5) are read from its
``reports/tables/hypotheses.json`` and compared per seed with ours (criteria: reports/planned-runs.md).
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
TABLE = "https://raw.githubusercontent.com/BurnyCoder/visualizing-grokking-modular-addition-in-transformer/main/reports/tables/hypotheses.json"
CURVES = ("loss_train", "loss_test", "restricted_loss", "excluded_loss", "gini_we", "ssw_total")
RUNS = (
    [f"wd1_s{i}" for i in range(11)]
    + [f"wd0_s{i}" for i in range(5)]
    + [f"rand_s{i}" for i in range(3)]
    + [f"split0_init{i}" for i in (100, 101, 102)]
)


def _get(run: str, name: str) -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(REPO, f"{run}/{name}")


def _at(steps: np.ndarray, t: int) -> int:
    """Index of the first checkpoint at or after step t (companion convention for per-checkpoint values "at t")."""
    return int(min(np.searchsorted(steps, t, side="left"), len(steps) - 1))


def progress(run: str, phases: dict) -> dict:
    """H1/H2 progress-measure tests on one run's per-checkpoint metrics + downsampled curves for the site."""
    z = np.load(_get(run, "analysis/ckpt_metrics.npz"))
    steps = z["steps"]
    i1 = _at(steps, phases["t1"])
    i2 = _at(steps, phases["t2"]) if phases.get("t2") else None

    def drop(series: np.ndarray) -> int | None:
        """Step of the first checkpoint >= t1 where the series is <= 10% of its value at t1 (None if never)."""
        hit = np.where(series[i1:] <= 0.1 * series[i1])[0]
        return int(steps[i1 + hit[0]]) if len(hit) else None

    t_r, t_te = drop(z["restricted_loss"]), drop(z["loss_test"])
    ratio = float(z["excluded_loss"][i2] / z["excluded_loss"][i1]) if i2 is not None else None
    sub = np.arange(0, len(steps), 2)  # every 2nd checkpoint (~200 points) keeps the site file small
    return {
        "t1": phases.get("t1"), "t2": phases.get("t2"), "t3": phases.get("t3"), "t_grok": phases.get("t_grok"),
        "t_restricted_drop": t_r, "t_test_drop": t_te,
        "h1": bool(t_r is not None and (t_te is None or t_r < t_te)),
        "h2_ratio": ratio, "h2": bool(ratio is not None and ratio > 10),
        "curves": {"step": steps[sub], **{k: z[k][sub] for k in CURVES}},
    }


def companion_table() -> dict:
    """The companion project's preregistered verdicts (reports/tables/hypotheses.json); {} if unreachable."""
    import requests

    try:
        return requests.get(TABLE, timeout=30).json()
    except Exception as e:  # network problems must not break the rest of the entry
        log.warning("companion hypotheses table unavailable: %s", e)
        return {}


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
        if meta["group"] == "wd1":
            pr = runs[-1]["progress"] = progress(r, phases)
            log.info("%s progress: restricted drop @%s vs test drop @%s (H1 %s); excluded ratio t2/t1 %s (H2 %s)",
                     r, pr["t_restricted_drop"], pr["t_test_drop"], pr["h1"], pr["h2_ratio"], pr["h2"])
        log.info("%s: key freqs %s, grok@%s, final test %.3f", r, runs[-1]["key_freqs"], runs[-1]["test95"], runs[-1]["final_test_acc"])

    wd1 = [r for r in runs if r["group"] == "wd1"]
    wd0 = [r for r in runs if r["group"] == "wd0"]
    rnd = [r for r in runs if r["group"] == "rand"]
    grokked = [r for r in wd1 if r["test95"] and r["final_test_acc"] > 0.99]
    delays = [r["test95"] / r["train99"] for r in grokked if r["train99"]]
    nkey = [len(r["key_freqs"]) for r in wd1]
    fracn = [r["frac_neurons_085"] for r in wd1 if r["frac_neurons_085"] is not None]
    prog = {r["id"]: r["progress"] for r in wd1}
    leads = [p["t_test_drop"] - p["t_restricted_drop"] for p in prog.values() if p["t_test_drop"] and p["t_restricted_drop"]]
    ratios = [p["h2_ratio"] for p in prog.values() if p["h2_ratio"]]
    table = companion_table()
    per_run, verdict = table.get("per_run", {}), table.get("verdict", {})
    # Agreement with the companion's per-seed numbers (same definitions, independent code).
    agree = sum(
        1 for rid, p in prog.items()
        if rid in per_run and per_run[rid].get("t_restricted_drop") == p["t_restricted_drop"] and per_run[rid].get("t_test_drop") == p["t_test_drop"]
    )

    def frac(h: str) -> str:
        v = verdict.get(h, {})
        return f"{v['successes']}/{v['n']}" if "successes" in v else "n/a"

    summary = {
        "n_runs": len(runs),
        "n_h1": int(sum(p["h1"] for p in prog.values())),
        "n_h2": int(sum(p["h2"] for p in prog.values())),
        "n_prog": len(prog),
        "h1_lead_min": int(min(leads)) if leads else -1,
        "h1_lead_max": int(max(leads)) if leads else -1,
        "h2_ratio_min": float(min(ratios)) if ratios else -1,
        "h2_ratio_max": float(max(ratios)) if ratios else -1,
        "agree_companion": agree,
        "comp_h3": frac("H3"),
        "comp_h4": frac("H4"),
        "comp_h4_supported": "yes" if verdict.get("H4", {}).get("supported") else "no",
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
        verdict_row(
            "Restricted loss falls before test loss (progress measure; companion H1, cross-check)",
            "restricted loss drops during circuit formation, before test loss (§5.1, Fig. 7); companion: 11/11 seeds",
            f"{summary['n_h1']}/{len(prog)} seeds; restricted loss reaches 10% of its t1 value {summary['h1_lead_min']}–{summary['h1_lead_max']} steps "
            f"before test loss; {agree}/{len(prog)} seeds match the companion's per-seed steps exactly",
            "earlier in ≥ 10 of 11 seeds",
            summary["n_h1"] >= len(prog) - 1 and len(prog) >= 10,
        ),
        verdict_row(
            "Excluded loss rises during circuit formation (progress measure; companion H2, cross-check)",
            "excluded loss rises while train and test loss barely move (§5.1); companion: 11/11 seeds",
            f"{summary['n_h2']}/{len(prog)} seeds; excluded loss at t2 is {summary['h2_ratio_min']:.3g}× to {summary['h2_ratio_max']:.3g}× its value at t1",
            "> 10× in ≥ 10 of 11 seeds",
            summary["n_h2"] >= len(prog) - 1 and len(prog) >= 10,
        ),
    ]
    payload = {"runs": runs, "source": f"https://huggingface.co/{REPO}", "companion_verdict": verdict, "companion_table": TABLE}
    write_entry(SLUG, payload, summary, repl, models=[REPO], runtime_s=time.time() - t0)
    log.info("grokking runs summary: %s", summary)
