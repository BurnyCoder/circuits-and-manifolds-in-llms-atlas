"""Small statistics helpers (bootstrap confidence intervals).

Local context: percentile bootstrap over examples (Efron & Tibshirani 1993); 95% CI by default. Used for every
per-prompt metric we report (logit differences, accuracies, scores) so the site can show error bars.
"""

from __future__ import annotations

import numpy as np


def bootstrap_ci(x, stat=np.mean, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float, float]:
    """Return (point estimate, low, high) of ``stat`` over the first axis of ``x`` via percentile bootstrap."""
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))  # resample examples with replacement
    boots = np.array([stat(x[i]) for i in idx])
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return float(stat(x)), float(lo), float(hi)


def ci_str(est: float, lo: float, hi: float, digits: int = 2, pct: bool = False) -> str:
    """Format '3.56 [3.40, 3.71]' (or percentages) for replication tables."""
    if pct:
        return f"{est * 100:.{digits - 1}f}% [{lo * 100:.{digits - 1}f}, {hi * 100:.{digits - 1}f}]"
    return f"{est:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"
