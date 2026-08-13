"""Classical position evaluation.

Evaluation is returned in centipawns from White's perspective:
positive means White is better, negative means Black is better.
Callers that need side-to-move-relative scores (e.g. negamax search)
should negate the value themselves when the side to move is Black.
"""

from __future__ import annotations

from typing import Dict

import chess

# Material values in centipawns.
PIECE_VALUES: Dict[chess.PieceType, int] = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,  # King safety handled separately; material value is 0.
}

# Piece-square tables, indexed by square 0..63 (a1=0 ... h8=63) from
# White's point of view. Black's score is computed by mirroring the
# square vertically. Values are in centipawns and encourage classical
# opening/middlegame piece placement.
_PAWN_TABLE = [
    0, 0, 0, 0, 0, 0, 0, 0,
    50, 50, 50, 50, 50, 50, 50, 50,
    10, 10, 20, 30, 30, 20, 10, 10,
    5, 5, 10, 25, 25, 10, 5, 5,
    0, 0, 0, 20, 20, 0, 0, 0,
    5, -5, -10, 0, 0, -10, -5, 5,
    5, 10, 10, -20, -20, 10, 10, 5,
    0, 0, 0, 0, 0, 0, 0, 0,
]

_KNIGHT_TABLE = [
    -50, -40, -30, -30, -30, -30, -40, -50,
    -40, -20, 0, 0, 0, 0, -20, -40,
    -30, 0, 10, 15, 15, 10, 0, -30,
    -30, 5, 15, 20, 20, 15, 5, -30,
    -30, 0, 15, 20, 20, 15, 0, -30,
    -30, 5, 10, 15, 15, 10, 5, -30,
    -40, -20, 0, 5, 5, 0, -20, -40,
    -50, -40, -30, -30, -30, -30, -40, -50,
]

_BISHOP_TABLE = [
    -20, -10, -10, -10, -10, -10, -10, -20,
    -10, 0, 0, 0, 0, 0, 0, -10,
    -10, 0, 5, 10, 10, 5, 0, -10,
    -10, 5, 5, 10, 10, 5, 5, -10,
    -10, 0, 10, 10, 10, 10, 0, -10,
    -10, 10, 10, 10, 10, 10, 10, -10,
    -10, 5, 0, 0, 0, 0, 5, -10,
    -20, -10, -10, -10, -10, -10, -10, -20,
]

_ROOK_TABLE = [
    0, 0, 0, 0, 0, 0, 0, 0,
    5, 10, 10, 10, 10, 10, 10, 5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    0, 0, 0, 5, 5, 0, 0, 0,
]

_QUEEN_TABLE = [
    -20, -10, -10, -5, -5, -10, -10, -20,
    -10, 0, 0, 0, 0, 0, 0, -10,
    -10, 0, 5, 5, 5, 5, 0, -10,
    -5, 0, 5, 5, 5, 5, 0, -5,
    0, 0, 5, 5, 5, 5, 0, -5,
    -10, 5, 5, 5, 5, 5, 0, -10,
    -10, 0, 5, 0, 0, 0, 0, -10,
    -20, -10, -10, -5, -5, -10, -10, -20,
]

_KING_MIDDLEGAME_TABLE = [
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -20, -30, -30, -40, -40, -30, -30, -20,
    -10, -20, -20, -20, -20, -20, -20, -10,
    20, 20, 0, 0, 0, 0, 20, 20,
    20, 30, 10, 0, 0, 10, 30, 20,
]

_PIECE_SQUARE_TABLES: Dict[chess.PieceType, list] = {
    chess.PAWN: _PAWN_TABLE,
    chess.KNIGHT: _KNIGHT_TABLE,
    chess.BISHOP: _BISHOP_TABLE,
    chess.ROOK: _ROOK_TABLE,
    chess.QUEEN: _QUEEN_TABLE,
    chess.KING: _KING_MIDDLEGAME_TABLE,
}

# Large but finite mate score. Kept well below int overflow concerns
# and far above any realistic material evaluation.
MATE_SCORE = 1_000_000


class Evaluator:
    """Computes a classical evaluation of a chess position.

    The total score is composed of:
        material_score + piece_square_score + mobility_score + king_safety_score

    All sub-scores are computed from White's perspective (positive
    favors White). Material dominates the total for this V1 evaluator.
    """

    def __init__(
        self,
        mobility_weight: float = 1.0,
        king_safety_weight: float = 1.0,
    ) -> None:
        """Create an evaluator.

        Args:
            mobility_weight: Scales the mobility component. Set to 0 to disable.
            king_safety_weight: Scales the king safety component. Set to 0 to disable.
        """
        self.mobility_weight = mobility_weight
        self.king_safety_weight = king_safety_weight

    def evaluate(self, board: chess.Board) -> int:
        """Evaluate a position.

        Args:
            board: A python-chess Board instance.

        Returns:
            Centipawn evaluation from White's perspective. Positive
            favors White, negative favors Black.
        """
        if board.is_checkmate():
            # Side to move is mated: bad for the side to move.
            return -MATE_SCORE if board.turn == chess.WHITE else MATE_SCORE
        if board.is_stalemate() or board.is_insufficient_material():
            return 0

        material = self._material_score(board)
        piece_square = self._piece_square_score(board)
        mobility = self._mobility_score(board) * self.mobility_weight
        king_safety = self._king_safety_score(board) * self.king_safety_weight

        return int(material + piece_square + mobility + king_safety)

    def _material_score(self, board: chess.Board) -> int:
        """Sum of piece values, White minus Black."""
        score = 0
        for piece_type, value in PIECE_VALUES.items():
            score += value * len(board.pieces(piece_type, chess.WHITE))
            score -= value * len(board.pieces(piece_type, chess.BLACK))
        return score

    def _piece_square_score(self, board: chess.Board) -> int:
        """Positional bonus/penalty based on piece placement."""
        score = 0
        for square, piece in board.piece_map().items():
            table = _PIECE_SQUARE_TABLES[piece.piece_type]
            if piece.color == chess.WHITE:
                score += table[square]
            else:
                # Mirror vertically for Black's perspective.
                mirrored = chess.square_mirror(square)
                score -= table[mirrored]
        return score

    def _mobility_score(self, board: chess.Board) -> int:
        """Difference in legal move counts, White minus Black.

        Uses a lightweight approximation: counts legal moves for the
        side to move directly, and pseudo-legal moves for the other
        side (an exact legal-move count for the non-moving side would
        require a costly board copy + null move).
        """
        side_to_move_moves = board.legal_moves.count()
        if board.turn == chess.WHITE:
            white_mobility = side_to_move_moves
            black_mobility = self._approximate_opponent_mobility(board)
        else:
            black_mobility = side_to_move_moves
            white_mobility = self._approximate_opponent_mobility(board)
        return white_mobility - black_mobility

    @staticmethod
    def _approximate_opponent_mobility(board: chess.Board) -> int:
        """Approximate mobility for the side NOT to move via pseudo-legal moves."""
        board_copy = board.copy(stack=False)
        board_copy.push(chess.Move.null())
        count = board_copy.legal_moves.count()
        return count

    def _king_safety_score(self, board: chess.Board) -> int:
        """Simple king safety heuristic based on pawn shield presence.

        Rewards having friendly pawns in front of a castled-looking
        king position. This is intentionally simple for V1.
        """
        score = 0
        for color in (chess.WHITE, chess.BLACK):
            king_square = board.king(color)
            if king_square is None:
                continue
            shield_bonus = self._pawn_shield_bonus(board, king_square, color)
            score += shield_bonus if color == chess.WHITE else -shield_bonus
        return score

    @staticmethod
    def _pawn_shield_bonus(board: chess.Board, king_square: chess.Square, color: chess.Color) -> int:
        """Count friendly pawns directly in front of the king (small bonus each)."""
        file = chess.square_file(king_square)
        rank = chess.square_rank(king_square)
        direction = 1 if color == chess.WHITE else -1
        bonus = 0
        for df in (-1, 0, 1):
            f = file + df
            if 0 <= f <= 7:
                shield_rank = rank + direction
                if 0 <= shield_rank <= 7:
                    square = chess.square(f, shield_rank)
                    piece = board.piece_at(square)
                    if piece is not None and piece.piece_type == chess.PAWN and piece.color == color:
                        bonus += 10
        return bonus
