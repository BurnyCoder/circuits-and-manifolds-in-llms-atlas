"""Ground-truth tests for the Fourier-probe helpers of the geometric-calculator entry (atlas.entries.calculator_probes)."""

import numpy as np

from atlas.entries.calculator_probes import PERIODS, peak_scores, probe_r2


def test_probe_recovers_planted_circle():
    """Activations that contain sin/cos(2π s/10) among 300 noise features: the probe's best period is 10 with R² ≈ 1,
    and periods the activations do not encode stay near R² = 0 (per-period ridge strength chosen on training rows)."""
    rng = np.random.default_rng(0)
    s = rng.integers(2, 199, 600)
    X = np.c_[np.sin(2 * np.pi * s / 10), np.cos(2 * np.pi * s / 10), rng.normal(size=(600, 300)) * 0.5]
    tr = rng.random(600) < 0.8
    r2 = probe_r2(X, s, tr)
    assert PERIODS[np.argmax(r2)] == 10
    assert r2[PERIODS == 10][0] > 0.9
    assert abs(np.median(r2)) < 0.1


def test_peak_score_ignores_smooth_trends():
    """A smooth rising R² curve (like long periods acting as a magnitude code) plus spikes at 10 and 50: the two spikes
    are the top peaks even though the long periods have higher raw R²."""
    r2 = np.linspace(0, 0.6, len(PERIODS))
    r2[PERIODS == 10] += 0.3
    r2[PERIODS == 50] += 0.3
    top = PERIODS[np.argsort(-peak_scores(r2))[:2]]
    assert sorted(top.tolist()) == [10, 50]
    assert PERIODS[np.argmax(r2)] != 10  # raw ranking would be fooled
