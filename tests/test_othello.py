"""Othello engine ground truth: opening moves, flips, and random games are legal and fill the board."""

import numpy as np

from atlas.common.othello import SQUARES, initial_board, legal_moves, play, random_game


def test_opening_has_four_moves_for_black():
    # Standard Othello: black has exactly 4 legal opening moves.
    assert sorted(legal_moves(initial_board(), 1)) == [19, 26, 37, 44]


def test_play_flips_disc():
    b = play(initial_board(), 19, 1)  # black plays (2,3) flipping white (3,3)
    assert b[3, 3] == 1 and b[2, 3] == 1
    assert (b != 0).sum() == 5


def test_random_games_are_legal():
    rng = np.random.default_rng(0)
    for _ in range(20):
        moves, boards, movers, _ = random_game(rng)
        assert len(set(moves)) == len(moves)
        assert all(m in SQUARES for m in moves)
        assert (boards[-1] != 0).sum() == 4 + len(moves)


def test_tokens_skip_center():
    assert len(SQUARES) == 60 and 27 not in SQUARES and 36 not in SQUARES
