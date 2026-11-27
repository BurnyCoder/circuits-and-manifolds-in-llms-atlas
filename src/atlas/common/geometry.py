"""Geometry helpers: PCA, circle/cyclic-order statistics, ridge probes.

Local context: small, dependency-light implementations (NumPy) so they are easy to test against synthetic ground
truth in tests/test_geometry.py. Used by the manifold entries (calendar circles, number helix, linebreak manifold,
colour bicone, belief-state fractal).
"""

from __future__ import annotations


import numpy as np


def pca(X: np.ndarray, k: int):
    """Return (scores [n, k], components [k, d], explained-variance ratios [k], mean [d]) via SVD of centered X."""
    mu = X.mean(0)
    U, S, Vt = np.linalg.svd(X - mu, full_matrices=False)
    var = S**2 / (S**2).sum()
    return (X - mu) @ Vt[:k].T, Vt[:k], var[:k], mu


def cyclic_order_score(angles: np.ndarray) -> float:
    """Fraction of consecutive classes (i, i+1 mod n) that are angular neighbours, maximized over direction.

    1.0 = classes appear around the circle in exactly their cyclic order (e.g. Mon, Tue, … , Sun).
    """
    n = len(angles)
    order = np.argsort(angles)  # classes sorted by angle around the circle
    pos = np.empty(n, dtype=int)
    pos[order] = np.arange(n)
    best = 0.0
    for sign in (1, -1):
        hits = sum(((pos[(i + 1) % n] - pos[i]) * sign) % n == 1 for i in range(n))
        best = max(best, hits / n)
    return best


def cyclic_order_pvalue(score: float, n: int, n_perm: int = 20000, seed: int = 0) -> float:
    """P(score >= observed) for uniformly random class arrangements on the circle (permutation null)."""
    rng = np.random.default_rng(seed)
    null = np.array([cyclic_order_score(rng.permutation(n).astype(float)) for _ in range(n_perm)])
    return float((null >= score - 1e-12).mean())


def ridge_r2(X_tr, y_tr, X_te, y_te, alpha: float = 1.0) -> float:
    """Closed-form ridge regression (features standardized on train) -> held-out R² for each target column, averaged."""
    mu, sd = X_tr.mean(0), X_tr.std(0) + 1e-6
    A, B = (X_tr - mu) / sd, (X_te - mu) / sd
    ym = y_tr.mean(0)
    W = np.linalg.solve(A.T @ A + alpha * np.eye(A.shape[1]), A.T @ (y_tr - ym))
    pred = B @ W + ym
    yt = y_te.reshape(len(y_te), -1)
    pr = pred.reshape(len(y_te), -1)
    ss_res = ((yt - pr) ** 2).sum(0)
    ss_tot = ((yt - yt.mean(0)) ** 2).sum(0)
    return float(np.mean(1 - ss_res / ss_tot))


def circular_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Jammalamadaka circular correlation between two angle arrays (radians)."""
    ma = np.angle(np.exp(1j * a).mean())
    mb = np.angle(np.exp(1j * b).mean())
    sa, sb = np.sin(a - ma), np.sin(b - mb)
    return float((sa * sb).sum() / np.sqrt((sa**2).sum() * (sb**2).sum()))


def best_rotation_flip(angles_pred: np.ndarray, angles_true: np.ndarray) -> float:
    """Mean absolute angular error after the best rotation and optional reflection (degrees)."""
    best = 1e9
    for s in (1, -1):
        d = np.angle(np.exp(1j * (s * angles_pred - angles_true)))
        off = np.angle(np.exp(1j * d).mean())
        err = np.abs(np.angle(np.exp(1j * (d - off)))).mean()
        best = min(best, err)
    return float(np.degrees(best))


