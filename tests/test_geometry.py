"""Ground-truth tests for geometry helpers: a perfect 7-point circle has cyclic order 1, shuffles score low, ridge
recovers a linear target, PCA of a planar circle puts all variance in 2 components."""

import numpy as np

from atlas.common.geometry import best_rotation_flip, circular_corr, cyclic_order_pvalue, cyclic_order_score, pca, ridge_r2


def test_cyclic_order_perfect_and_reflected():
    ang = 2 * np.pi * np.arange(7) / 7
    assert cyclic_order_score(ang) == 1.0
    assert cyclic_order_score(-ang) == 1.0  # reflection is allowed


def test_cyclic_order_shuffled_is_rarely_perfect():
    p = cyclic_order_pvalue(1.0, 7, n_perm=5000)
    assert p < 0.01  # 2 of 720 arrangements up to rotation are perfect


def test_pca_circle_plane():
    t = np.linspace(0, 2 * np.pi, 50, endpoint=False)
    X = np.stack([np.cos(t), np.sin(t), np.zeros_like(t), np.zeros_like(t)], 1) @ np.linalg.qr(np.random.default_rng(0).normal(size=(4, 4)))[0]
    _, _, var, _ = pca(X, 3)
    assert var[:2].sum() > 0.999


def test_ridge_recovers_linear_target():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 10))
    y = X @ rng.normal(size=10)
    assert ridge_r2(X[:300], y[:300, None], X[300:], y[300:, None], alpha=1e-3) > 0.99


def test_circular_stats():
    a = np.linspace(0, 2 * np.pi, 30, endpoint=False)
    assert circular_corr(a, a) > 0.99
    assert best_rotation_flip(-(a + 1.0), a) < 1e-6


def test_ridge_alpha_selection_beats_overfit_alpha():
    """Alpha chosen on an inner grouped validation split of the training rows scores close to the best alpha on the
    grid (evaluated on the test set) and far above a badly chosen one."""
    rng = np.random.default_rng(1)
    n, d = 300, 1000
    X = rng.normal(size=(n, d))
    y = X[:, :20].sum(1) + rng.normal(scale=2.0, size=n)  # signal in 20 of 1000 features, noisy
    groups = np.arange(n) // 10  # "documents" of 10 rows
    tr, te = groups % 5 != 0, groups % 5 == 0
    grid = [1e-2, 1e0, 1e2, 1e3, 1e4, 1e5]
    fixed = {a: ridge_r2(X[tr], y[tr, None], X[te], y[te, None], alpha=a) for a in grid}
    cv = ridge_r2(X[tr], y[tr, None], X[te], y[te, None], alpha=grid, groups=groups[tr])
    assert cv >= max(fixed.values()) - 0.05  # the inner split picks a near-best alpha without seeing the test set
    assert cv > fixed[1e5] + 0.1  # and avoids a badly over-regularized one
