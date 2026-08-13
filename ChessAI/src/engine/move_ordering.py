"""Move ordering heuristics to improve alpha-beta pruning efficiency.

Good move ordering causes more beta cutoffs earlier in the search,
which is the single biggest practical speedup for alpha-beta search.
This module is intentionally simple for V1: captures, promotions,
and checks are searched first. More advanced ordering (killer moves,
history heuristic, MVV-LVA refinement, PV-move-first) can be added
later without changing the public interface.
"""

from __future__ import annotations

from typing import List, Optional

import chess

# MVV-LVA-ish victim values used to prioritize capturing higher-value
# pieces with lower-value pieces first.
_PIECE_ORDER_VALUE = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 20,
}


def order_moves(
    board: chess.Board,
    moves: List[chess.Move],
    tt_move: Optional[chess.Move] = None,
) -> List[chess.Move]:
    """Return moves sorted with the most promising ones first.

    Priority (highest first):
        1. Transposition-table best move (if provided and present).
        2. Captures, ordered by MVV-LVA (most valuable victim,
           least valuable attacker).
        3. Promotions.
        4. Checks.
        5. All other moves, unchanged relative order.

    Args:
        board: Current position (moves are pseudo-evaluated against this).
        moves: Legal moves to order.
        tt_move: Optional best move retrieved from the transposition table.

    Returns:
        A new list containing the same moves, reordered.
    """

    def score(move: chess.Move) -> int:
        if tt_move is not None and move == tt_move:
            return 1_000_000

        s = 0
        if board.is_capture(move):
            victim = _victim_value(board, move)
            attacker_piece = board.piece_at(move.from_square)
            attacker_value = (
                _PIECE_ORDER_VALUE.get(attacker_piece.piece_type, 0)
                if attacker_piece is not None
                else 0
            )
            # Prioritize high-value victims, then low-value attackers.
            s += 100_000 + victim * 10 - attacker_value

        if move.promotion is not None:
            s += 50_000 + _PIECE_ORDER_VALUE.get(move.promotion, 0) * 10

        if board.gives_check(move):
            s += 10_000

        return s

    return sorted(moves, key=score, reverse=True)


def _victim_value(board: chess.Board, move: chess.Move) -> int:
    """Return the piece-order value of the piece being captured by `move`."""
    if board.is_en_passant(move):
        return _PIECE_ORDER_VALUE[chess.PAWN]
    victim = board.piece_at(move.to_square)
    if victim is None:
        return 0
    return _PIECE_ORDER_VALUE.get(victim.piece_type, 0)
