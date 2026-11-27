"""Ground-truth tests for head scores: a hand-built 'perfect' induction / previous-token pattern must score 1."""

import torch

from atlas.common.attention import head_scores, repeated_tokens


def _pattern_for(offset: int, n: int, start: int) -> torch.Tensor:
    """[1, 1, n, n] pattern attending with weight 1 to dest-offset for dest>=start, else to BOS."""
    p = torch.zeros(1, 1, n, n)
    for i in range(n):
        p[0, 0, i, i - offset if i >= start else 0] = 1.0
    return p


def test_repeated_tokens_shape_and_repeat():
    t = repeated_tokens(batch=3, seq_len=5, vocab_lo=10, vocab_hi=20, bos=0)
    assert t.shape == (3, 11)
    assert torch.equal(t[:, 1:6], t[:, 6:11])
    assert (t[:, 0] == 0).all()


def test_perfect_induction_head_scores_one():
    L = 6
    n = 1 + 2 * L
    s = head_scores(_pattern_for(L - 1, n, 1 + L), L)
    assert torch.isclose(s["induction"], torch.tensor([1.0])).all()
    assert s["duplicate"].item() == 0.0


def test_perfect_previous_token_head_scores_one():
    L = 6
    s = head_scores(_pattern_for(1, 1 + 2 * L, 1), L)
    assert torch.isclose(s["prev"], torch.tensor([1.0])).all()
    assert s["induction"].item() == 0.0
