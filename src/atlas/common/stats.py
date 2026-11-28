"""Small statistics helpers (bootstrap and Wilson confidence intervals).

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


def wilson_ci(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """95% Wilson score interval for a rate k/n (Wilson 1927; same formula as statsmodels proportion_confint "wilson").

    centre = (p + z²/2n) / (1 + z²/n), half-width = z / (1 + z²/n) · sqrt(p(1−p)/n + z²/4n²). Unlike the normal
    approximation it stays inside [0, 1] and is reliable for small n and rates near 0 or 1 (Brown, Cai & DasGupta 2001).
    """
    if n == 0:
        return 0.0, 1.0
    p, z2 = k / n, z * z
    centre = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = z / (1 + z2 / n) * np.sqrt(p * (1 - p) / n + z2 / (4 * n * n))
    return float(max(0.0, centre - half)), float(min(1.0, centre + half))


def rate_str(k: int, n: int) -> str:
    """Format a rate with its Wilson 95% CI in percent, e.g. '30% [19, 45]'."""
    lo, hi = wilson_ci(k, n)
    return f"{100 * k / max(n, 1):.0f}% [{100 * lo:.0f}, {100 * hi:.0f}]"


def ci_str(est: float, lo: float, hi: float, digits: int = 2, pct: bool = False) -> str:
    """Format '3.56 [3.40, 3.71]' (or percentages) for replication tables."""
    if pct:
        return f"{est * 100:.{digits - 1}f}% [{lo * 100:.{digits - 1}f}, {hi * 100:.{digits - 1}f}]"
    return f"{est:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"
