"""Board abstraction wrapping python-chess.

This module isolates the rest of the engine from the underlying
chess-rules library. All other modules should interact with the
board through this class rather than importing `chess` directly
where practical.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import chess

logger = logging.getLogger(__name__)


class Board:
    """Wraps a python-chess Board and exposes a clean interface."""

    def __init__(self, fen: Optional[str] = None) -> None:
        """Create a new board.

        Args:
            fen: Optional FEN string to load. If None, starts a new game.
        """
        self._board: chess.Board = chess.Board(fen) if fen else chess.Board()

    @classmethod
    def new_game(cls) -> "Board":
        """Create a board in the standard starting position."""
        return cls()

    @classmethod
    def from_fen(cls, fen: str) -> "Board":
        """Create a board from a FEN string."""
        return cls(fen=fen)

    @classmethod
    def from_raw(cls, raw: chess.Board) -> "Board":
        """Wrap an existing python-chess Board without copying.

        Intended for agents that already hold a raw ``chess.Board`` and want
        to reuse the wrapped interface (e.g. the classical engine agent).
        """
        wrapper = cls()
        wrapper._board = raw
        return wrapper

    def make_move(self, move: chess.Move) -> None:
        """Apply a move to the board.

        Args:
            move: A legal chess.Move to apply.

        Raises:
            ValueError: If the move is not legal in the current position.
        """
        if move not in self._board.legal_moves:
            raise ValueError(f"Illegal move attempted: {move}")
        self._board.push(move)

    def make_move_uci(self, uci: str) -> None:
        """Apply a move given in UCI notation (e.g. 'e2e4').

        Args:
            uci: Move string in UCI format.

        Raises:
            ValueError: If the move string is invalid or illegal.
        """
        try:
            move = chess.Move.from_uci(uci)
        except ValueError as exc:
            raise ValueError(f"Invalid UCI move string: {uci}") from exc
        self.make_move(move)

    def undo_move(self) -> Optional[chess.Move]:
        """Undo the last move.

        Returns:
            The move that was undone, or None if there was no move to undo.
        """
        if not self._board.move_stack:
            logger.debug("Undo requested but no moves to undo.")
            return None
        return self._board.pop()

    def legal_moves(self) -> List[chess.Move]:
        """Return the list of legal moves in the current position."""
        return list(self._board.legal_moves)

    def is_legal(self, move: chess.Move) -> bool:
        """Check whether a move is legal in the current position."""
        return move in self._board.legal_moves

    def is_check(self) -> bool:
        """Return True if the side to move is in check."""
        return self._board.is_check()

    def is_checkmate(self) -> bool:
        """Return True if the current position is checkmate."""
        return self._board.is_checkmate()

    def is_stalemate(self) -> bool:
        """Return True if the current position is stalemate."""
        return self._board.is_stalemate()

    def is_insufficient_material(self) -> bool:
        """Return True if neither side has enough material to mate."""
        return self._board.is_insufficient_material()

    def is_draw(self) -> bool:
        """Return True if the position is a draw by any standard rule.

        Covers stalemate, insufficient material, seventy-five-move rule,
        fivefold repetition, and claimable draws (fifty-move / threefold).
        """
        return (
            self._board.is_stalemate()
            or self._board.is_insufficient_material()
            or self._board.is_seventyfive_moves()
            or self._board.is_fivefold_repetition()
            or self._board.can_claim_draw()
        )

    def is_game_over(self) -> bool:
        """Return True if the game has ended (checkmate, stalemate, or draw)."""
        return self._board.is_game_over()

    def result(self) -> str:
        """Return the game result string, e.g. '1-0', '0-1', '1/2-1/2', '*'."""
        return self._board.result(claim_draw=True)

    def get_fen(self) -> str:
        """Return the current position as a FEN string."""
        return self._board.fen()

    def get_move_history(self) -> List[str]:
        """Return the move history as a list of UCI strings."""
        return [move.uci() for move in self._board.move_stack]

    def get_san_history(self) -> List[str]:
        """Return the move history in Standard Algebraic Notation.

        Note: This replays the game internally since SAN depends on
        board context at the time each move was made.
        """
        temp_board = chess.Board()
        san_moves: List[str] = []
        for move in self._board.move_stack:
            san_moves.append(temp_board.san(move))
            temp_board.push(move)
        return san_moves

    def turn(self) -> chess.Color:
        """Return the color to move (chess.WHITE or chess.BLACK)."""
        return self._board.turn

    def piece_at(self, square: chess.Square) -> Optional[chess.Piece]:
        """Return the piece at a given square, or None if empty."""
        return self._board.piece_at(square)

    def fullmove_number(self) -> int:
        """Return the current full move number."""
        return self._board.fullmove_number

    def copy(self) -> "Board":
        """Return a deep copy of this board."""
        new_board = Board.__new__(Board)
        new_board._board = self._board.copy()
        return new_board

    @property
    def raw(self) -> chess.Board:
        """Access the underlying python-chess Board directly.

        Provided as an escape hatch for modules (evaluator, search)
        that need direct access to python-chess functionality not
        wrapped here. Prefer the wrapped methods where possible.
        """
        return self._board

    def __str__(self) -> str:
        return str(self._board)
