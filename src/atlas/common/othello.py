"""Minimal Othello engine matching the synthetic dataset of Othello-GPT (Li et al. 2023).

Local context: the board is an 8x8 array, 1 = black, -1 = white, 0 = empty; initial position and "black moves first"
follow ``OthelloBoardState`` in github.com/likenneth/othello_world (data/othello.py): black on (3,4) and (4,3), white on
(3,3) and (4,4). A player with no legal move passes (passes are not tokens). Square index = row*8 + col.
Tokenization (TransformerLens ``othello-gpt``): the 60 non-center squares in row-major order get tokens 1..60
(token 0 is unused/pass), i.e. squares 27, 28, 35, 36 are skipped — as in Neel Nanda's Othello-GPT utilities.
"""

from __future__ import annotations

import numpy as np

DIRS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
CENTER = {27, 28, 35, 36}
SQUARES = [i for i in range(64) if i not in CENTER]  # token t (1..60) <-> SQUARES[t-1]
SQ_TO_TOK = {s: i + 1 for i, s in enumerate(SQUARES)}


def initial_board() -> np.ndarray:
    b = np.zeros((8, 8), dtype=np.int8)
    b[3, 4] = b[4, 3] = 1
    b[3, 3] = b[4, 4] = -1
    return b


def flips(board: np.ndarray, r: int, c: int, color: int) -> list[tuple[int, int]]:
    """Discs flipped if ``color`` plays at (r, c); empty list = illegal."""
    if board[r, c] != 0:
        return []
    out = []
    for dr, dc in DIRS:
        rr, cc, line = r + dr, c + dc, []
        while 0 <= rr < 8 and 0 <= cc < 8 and board[rr, cc] == -color:
            line.append((rr, cc))
            rr += dr
            cc += dc
        if line and 0 <= rr < 8 and 0 <= cc < 8 and board[rr, cc] == color:
            out += line
    return out


def legal_moves(board: np.ndarray, color: int) -> list[int]:
    return [r * 8 + c for r in range(8) for c in range(8) if flips(board, r, c, color)]


def play(board: np.ndarray, sq: int, color: int) -> np.ndarray:
    r, c = divmod(sq, 8)
    f = flips(board, r, c, color)
    assert f, f"illegal move {sq} for {color}"
    b = board.copy()
    b[r, c] = color
    for rr, cc in f:
        b[rr, cc] = color
    return b


def random_game(rng: np.random.Generator):
    """Uniformly random legal game. Returns (moves, boards_after, movers, next_legal) — one entry per move."""
    board, color = initial_board(), 1
    moves, boards, movers, next_legal = [], [], [], []
    while True:
        legal = legal_moves(board, color)
        if not legal:
            color = -color
            legal = legal_moves(board, color)
            if not legal:
                break
        sq = int(rng.choice(legal))
        board = play(board, sq, color)
        moves.append(sq)
        boards.append(board.copy())
        movers.append(color)
        # Who moves next (handles passes) and their legal moves: what the model must predict after this move.
        nxt = -color if legal_moves(board, -color) else color
        next_legal.append(legal_moves(board, nxt))
        color = -color
    return moves, boards, movers, next_legal


def next_player(board: np.ndarray, mover: int) -> int:
    """Color to move after ``mover`` played (opponent unless they must pass)."""
    return -mover if legal_moves(board, -mover) else mover
