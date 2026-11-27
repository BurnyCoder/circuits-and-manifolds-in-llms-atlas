"""Attention-pattern scores used to classify heads (previous-token, induction, duplicate-token).

Local context: on a sequence ``[BOS, t_1..t_L, t_1..t_L]`` (random tokens repeated once), position ``i`` in the
second half sees its own token earlier at ``i - L`` (duplicate-token offset) and the token that *followed* it at
``i - L + 1`` (induction offset). A head's score is its mean attention weight along that diagonal of the pattern,
averaged over the second half and the batch — the standard behavioural definition of Olsson et al. (2022,
https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html#definition-of-induction-heads)
as implemented in ARENA's induction-head exercises (https://arena-chapter1-transformer-interp.streamlit.app/).
Global context: these scores color the GPT-2 Model Map and identify heads in the induction entry.
"""

from __future__ import annotations

import torch


def repeated_tokens(batch: int, seq_len: int, vocab_lo: int, vocab_hi: int, bos: int, seed: int = 0) -> torch.Tensor:
    """[batch, 1 + 2*seq_len] tensor: BOS followed by a random token block repeated twice."""
    g = torch.Generator().manual_seed(seed)
    half = torch.randint(vocab_lo, vocab_hi, (batch, seq_len), generator=g)
    return torch.cat([torch.full((batch, 1), bos), half, half], dim=1)


def diag_score(pattern: torch.Tensor, offset: int, start: int) -> torch.Tensor:
    """Mean attention along the diagonal ``dest - src = offset`` for destination positions >= ``start``.

    ``pattern``: [batch, heads, dest, src]. Returns [heads].
    """
    d = pattern.diagonal(offset=-offset, dim1=-2, dim2=-1)  # [batch, heads, n_dest - offset]; d[..., k] = p[k+offset, k]
    first_dest = offset  # diagonal element k has dest index k + offset
    return d[..., max(0, start - first_dest) :].mean(dim=(0, -1))


def head_scores(pattern: torch.Tensor, seq_len: int) -> dict[str, torch.Tensor]:
    """Previous-token, induction and duplicate-token scores for every head of one layer on repeated tokens."""
    second_half = 1 + seq_len  # index of the first token of the repeat
    return {
        "prev": diag_score(pattern, 1, 1),
        "induction": diag_score(pattern, seq_len - 1, second_half),
        "duplicate": diag_score(pattern, seq_len, second_half),
    }
