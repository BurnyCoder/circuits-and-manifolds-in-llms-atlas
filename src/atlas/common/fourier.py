"""Fourier basis over Z_p, as used in Nanda et al. 2023 (https://arxiv.org/abs/2301.05217, §4).

Local context: rows are the constant function, then cos(2πkx/p), sin(2πkx/p) for k = 1..(p-1)/2, each normalized to
unit L2 norm, so the basis is orthonormal for odd p. ``F @ W`` expresses a [p, d] matrix (e.g. the token embedding)
in the Fourier basis along the token axis. Matches the ARENA grokking exercises' ``make_fourier_basis``.
"""

from __future__ import annotations

import torch


def fourier_basis(p: int, device="cpu") -> tuple[torch.Tensor, list[str]]:
    """[p, p] orthonormal basis (rows) and their names ('const', 'cos 1', 'sin 1', ...)."""
    x = torch.arange(p, dtype=torch.float64, device=device)
    rows, names = [torch.ones(p, dtype=torch.float64, device=device)], ["const"]
    for k in range(1, p // 2 + 1):
        rows += [torch.cos(2 * torch.pi * k * x / p), torch.sin(2 * torch.pi * k * x / p)]
        names += [f"cos {k}", f"sin {k}"]
    F = torch.stack(rows)
    return F / F.norm(dim=1, keepdim=True), names


def freq_norms(W: torch.Tensor, F: torch.Tensor) -> torch.Tensor:
    """L2 norm of a [p, d] matrix per frequency k = 1..(p-1)/2, combining the cos and sin rows: [(p-1)/2]."""
    c = (F.to(W.dtype) @ W).norm(dim=1)  # [p] norms of each basis component
    return torch.sqrt(c[1::2] ** 2 + c[2::2] ** 2)


def project_out_freqs(W: torch.Tensor, F: torch.Tensor, freqs: list[int]) -> torch.Tensor:
    """Remove the cos/sin components of the given frequencies from a [p, d] matrix (token axis)."""
    rows = [2 * k - 1 for k in freqs] + [2 * k for k in freqs]
    B = F[rows].to(W.dtype)  # [2m, p]
    return W - B.T @ (B @ W)
