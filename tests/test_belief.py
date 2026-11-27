"""Mess3 ground truth: valid stochastic matrices, beliefs stay on the simplex, and sampling matches the HMM."""

import numpy as np

from atlas.entries.belief_fractal import beliefs, mess3, sample, simplex_xy


def test_mess3_is_stochastic():
    T = mess3()
    assert T.shape == (3, 3, 3)
    assert np.allclose(T.sum((0, 2)), 1)
    assert (T >= 0).all()


def test_beliefs_on_simplex():
    T = mess3()
    s = sample(T, 200, np.random.default_rng(0))
    b = beliefs(T, s)
    assert np.allclose(b.sum(-1), 1) and (b >= 0).all()
    xy = simplex_xy(b)
    assert xy[..., 1].min() >= -1e-9 and xy[..., 1].max() <= np.sqrt(3) / 2 + 1e-9


def test_token_frequencies_uniform():
    # Mess3 is symmetric, so tokens are equally frequent in the stationary process.
    s = sample(mess3(), 20000, np.random.default_rng(1))
    freq = np.bincount(s.ravel(), minlength=3) / s.size
    assert np.allclose(freq, 1 / 3, atol=0.01)
