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
    if len(np.unique(np.round(angles, 9))) < n:
        return float("nan")  # tied angles have no order (a stable argsort would score all-equal angles as 1.0)
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
    if not np.isfinite(score):
        return 1.0  # tied angles (NaN score): never significant
    rng = np.random.default_rng(seed)
    null = np.array([cyclic_order_score(rng.permutation(n).astype(float)) for _ in range(n_perm)])
    return float((null >= score - 1e-12).mean())


def ridge_r2(X_tr, y_tr, X_te, y_te, alpha: float | list[float] = 1.0, groups=None) -> float:
    """Closed-form ridge regression (features standardized on train) -> held-out R² for each target column, averaged.

    If ``alpha`` is a list, it is chosen by held-out R² on an inner validation split of the *training* rows (every 5th
    group if ``groups`` is given, e.g. document ids, so tokens of one document never sit on both sides; else every 5th
    row), then the model is refit on all training rows — the same idea as sklearn's RidgeCV
    (https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeCV.html). The test set is never used
    for the choice.
    """
    return r2_score(y_te, ridge_predict(X_tr, y_tr, X_te, alpha, groups))


def ridge_predict(X_tr, y_tr, X_te, alpha: float | list[float] = 1.0, groups=None) -> np.ndarray:
    """Ridge predictions for X_te (same fitting and inner-split alpha choice as ``ridge_r2``)."""
    if np.ndim(alpha) > 0:
        g = np.asarray(groups) if groups is not None else np.arange(len(X_tr))
        uniq = np.unique(g)
        val = np.isin(g, uniq[::5])
        alpha = max(alpha, key=lambda a: ridge_r2(X_tr[~val], y_tr[~val], X_tr[val], y_tr[val], float(a)))
    mu, sd = X_tr.mean(0), X_tr.std(0) + 1e-6
    A, B = (X_tr - mu) / sd, (X_te - mu) / sd
    ym = y_tr.mean(0)
    W = np.linalg.solve(A.T @ A + alpha * np.eye(A.shape[1]), A.T @ (y_tr - ym))
    return B @ W + ym


def r2_score(y, pred) -> float:
    """R² per target column (1 − SS_res / SS_tot around the column mean of y), averaged over columns."""
    yt = np.asarray(y).reshape(len(y), -1)
    pr = np.asarray(pred).reshape(len(y), -1)
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




def validated_probe(X: dict, y, groups, alphas, seed: int = 0, early_max: int | None = None, y_shuf=None) -> dict:
    """Confirmatory linear probe with a layer chosen on validation groups (review follow-ups, methods review M9).

    Test groups: id % 5 == 0. Validation groups: every 5th remaining group. The layer with the best validation R² (ridge
    fit on the other training groups, alpha chosen inside) is refit on all training groups and scored on the test
    groups, with a group-bootstrap 95% CI. Optional: the shuffled-label R² at that layer (``y_shuf``) and the ratio
    R²(best validation layer ≤ ``early_max``) / R²(chosen layer) with its CI from the same resamples.
    """
    from .stats import group_bootstrap

    groups = np.asarray(groups)
    test = groups % 5 == 0
    val = np.isin(groups, np.unique(groups[~test])[::5])
    fit = ~test & ~val
    val_r2 = {L: r2_score(y[val], ridge_predict(Xl[fit], y[fit], Xl[val], alphas, groups[fit])) for L, Xl in X.items()}
    L_best = max(val_r2, key=val_r2.get)
    layers = {L_best}
    if early_max is not None:
        L_early = max((L for L in val_r2 if L <= early_max), key=val_r2.get)
        layers.add(L_early)
    pred = {L: ridge_predict(X[L][~test], y[~test], X[L][test], alphas, groups[~test]) for L in layers}
    yt, g = y[test], groups[test]
    r2 = r2_score(yt, pred[L_best])
    lo, hi = group_bootstrap(g, lambda i: r2_score(yt[i], pred[L_best][i]), seed=seed)
    out = {"layer": L_best, "val_r2": val_r2, "r2": r2, "r2_lo": lo, "r2_hi": hi, "n_test_groups": int(len(np.unique(g))),
           "pred": pred[L_best], "y_test": yt, "test_groups": g}
    if y_shuf is not None:
        out["r2_shuffled"] = r2_score(y_shuf[test], ridge_predict(X[L_best][~test], y_shuf[~test], X[L_best][test], alphas, groups[~test]))
    if early_max is not None:
        r2e = r2_score(yt, pred[L_early])
        rlo, rhi = group_bootstrap(g, lambda i: r2_score(yt[i], pred[L_early][i]) / r2_score(yt[i], pred[L_best][i]), seed=seed)
        out.update({"early_layer": L_early, "r2_early": r2e, "ratio": r2e / r2, "ratio_lo": rlo, "ratio_hi": rhi})
    return out
