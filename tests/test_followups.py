"""Ground-truth tests for the helpers added by the methods-review follow-ups (reports/planned-runs.md, 2026-11-28).

Each test plants a known answer: an interior R² peak, a polyline through known centroids, a single labelled path in a
toy attribution graph, a probe signal in one layer, and fixed IOI token positions.
"""

from types import SimpleNamespace

import numpy as np
import torch

from atlas.common.geometry import validated_probe
from atlas.entries.attribution_graphs import label_null_p
from atlas.entries.calculator_probes import PERIODS, valid_peak_hits
from atlas.entries.manifold_steering import polyline
from atlas.entries.sae_manifolds import peak_test


def test_peak_test_finds_interior_peak_and_dilution_rise():
    # 6 seeds × 4 k values × 10 instances; R² peaks at the 2nd k; dilution only at the last k.
    rng = np.random.default_rng(0)
    R2 = np.array([0.4, 0.6, 0.5, 0.3])[None, :, None] + 0.01 * rng.normal(size=(6, 4, 10))
    REG = np.full((6, 4, 10), "capture", dtype=object)
    REG[:, 0] = "shattering"
    REG[:, -1] = "dilution"
    pk = peak_test(R2, REG, 3, seed=0, n_boot=200)
    assert pk["k_star_idx"] == 1
    assert abs(pk["delta"] - 0.2) < 0.02 and pk["delta_lo"] > 0
    assert pk["dil_rise"] == 1.0 and pk["shat_drop"] == 1.0


def test_polyline_visits_intermediate_centroids():
    C = np.eye(7)  # day centroids at the 7 unit vectors
    ts = np.linspace(0, 1, 13)
    P = polyline(C, 5, 3, ts)  # 5 -> 6 -> 0 -> 1 (wraps around the week)
    assert np.allclose(P[0], C[5]) and np.allclose(P[4], C[6]) and np.allclose(P[8], C[0]) and np.allclose(P[-1], C[1])
    P2 = polyline(C, 2, -2, ts)  # 2 -> 1 -> 0, intermediate reached at t = 0.5
    assert np.allclose(P2[6], C[1]) and np.allclose(P2[-1], C[0])


def test_valid_peak_hits_drops_near_zero_r2():
    r2 = np.full(len(PERIODS), -0.05)
    r2[PERIODS == 10] = 0.3  # a real peak
    r2[PERIODS == 5] = 0.03  # sticks out of its negative neighbours, but below the R² floor
    hits = valid_peak_hits(r2)
    assert hits == [10]


def test_label_null_on_a_single_labelled_path():
    # Dallas → Texas → Austin is the only path; 5 other features are unconnected. Permuting the 6 feature labels puts
    # "Texas" on the true node 1 time in 6, so p ≈ 1/6.
    nodes = [{"id": "e", "label": "Dallas", "kind": "embed"}, {"id": "t", "label": "Texas", "kind": "feature"}]
    nodes += [{"id": f"f{i}", "label": f"other {i}", "kind": "feature"} for i in range(5)]
    nodes += [{"id": "l", "label": "→ Austin", "kind": "logit"}]
    s = {"nodes": nodes, "edges": [{"s": "e", "t": "t", "w": 1.0}, {"s": "t", "t": "l", "w": 1.0}]}
    p = label_null_p(s, "Dallas", "Texas", "Austin", n_perm=600, seed=0)
    assert abs(p - 1 / 6) < 0.05


def test_validated_probe_picks_the_signal_layer():
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(50), 20)  # 50 documents × 20 tokens
    y = rng.normal(size=(1000, 1))
    X = {0: rng.normal(size=(1000, 8)), 3: np.c_[y + 0.1 * rng.normal(size=(1000, 1)), rng.normal(size=(1000, 7))]}
    out = validated_probe(X, y, groups, [1e-2, 1e0, 1e2], seed=0, early_max=0, y_shuf=rng.permutation(y))
    assert out["layer"] == 3 and out["r2"] > 0.95 and out["r2_lo"] <= out["r2"] <= out["r2_hi"]
    assert out["r2_shuffled"] < 0.1 and out["ratio"] < 0.1


def test_ioi_node_mask_positions():
    from atlas.entries.ioi import CLASSES, node_mask

    model = SimpleNamespace(cfg=SimpleNamespace(n_layers=12, n_heads=12))
    toks = torch.tensor([[50256, 7, 9, 5, 7, 3, 4]])  # subject token 7 at positions 1 (S1) and 4 (S2); END = 6
    keep = node_mask(model, toks, torch.tensor([7]))
    l, h = CLASSES["Previous token"][0]
    assert keep[0, 2, l, h] and keep[0].sum(0)[l, h] == 1  # S1 + 1
    l, h = CLASSES["Induction"][0]
    assert keep[0, 4, l, h]
    l, h = CLASSES["Name mover"][0]
    assert keep[0, 6, l, h]
    assert keep.sum() == sum(len(v) for v in CLASSES.values())  # one position per circuit head
