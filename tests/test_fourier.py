"""Fourier basis ground truth: orthonormal, and a pure cos(2π·7x/p) embedding has all its norm at k=7."""

import torch

from atlas.common.fourier import fourier_basis, freq_norms, project_out_freqs


def test_basis_orthonormal():
    F, names = fourier_basis(13)
    assert torch.allclose(F @ F.T, torch.eye(13, dtype=F.dtype), atol=1e-10)
    assert names[:3] == ["const", "cos 1", "sin 1"]


def test_single_frequency_embedding():
    p = 113
    F, _ = fourier_basis(p)
    x = torch.arange(p, dtype=torch.float64)
    W = torch.stack([torch.cos(2 * torch.pi * 7 * x / p), torch.sin(2 * torch.pi * 7 * x / p)], 1)  # [p, 2]
    n = freq_norms(W, F)
    assert n.argmax().item() == 6  # index k-1
    assert n[6] / n.sum() > 0.999
    assert project_out_freqs(W, F, [7]).abs().max() < 1e-10
