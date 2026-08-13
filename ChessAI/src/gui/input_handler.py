"""Translates raw PyGame mouse events into chess move selections.

Kept independent of rendering and search: this module only tracks
click state (selected square) and produces a chess.Move once a
from/to pair is chosen, handling underpromotion by defaulting to queen.
"""

from __future__ import annotations

from typing import List, Optional

import chess

from src.gui.board_renderer import BoardRenderer


class InputHandler:
    """Tracks square selection state and builds moves from clicks."""

    def __init__(self, renderer: BoardRenderer) -> None:
        """Create an input handler bound to a renderer for coordinate conversion."""
        self.renderer = renderer
        self.selected_square: Optional[chess.Square] = None

    def legal_targets(self, board: chess.Board) -> List[chess.Square]:
        """Return legal destination squares for the currently selected piece."""
        if self.selected_square is None:
            return []
        return [
            move.to_square
            for move in board.legal_moves
            if move.from_square == self.selected_square
        ]

    def handle_click(self, board: chess.Board, pixel_x: int, pixel_y: int) -> Optional[chess.Move]:
        """Process a mouse click and return a move if one is completed.

        Args:
            board: Current position, used to validate selections and moves.
            pixel_x: X pixel coordinate of the click.
            pixel_y: Y pixel coordinate of the click.

        Returns:
            A legal chess.Move if this click completed a move selection,
            otherwise None (e.g. the click just selected a piece, selected
            an empty/invalid square, or deselected the current piece).
        """
        square = self.renderer.pixel_to_square(pixel_x, pixel_y)
        if square is None:
            return None

        if self.selected_square is None:
            return self._try_select(board, square)

        if square == self.selected_square:
            self.selected_square = None
            return None

        move = self._build_move(board, self.selected_square, square)
        if move is not None and move in board.legal_moves:
            self.selected_square = None
            return move

        # Clicked another one of our own pieces: switch selection.
        return self._try_select(board, square)

    def _try_select(self, board: chess.Board, square: chess.Square) -> None:
        piece = board.piece_at(square)
        if piece is not None and piece.color == board.turn:
            self.selected_square = square
        else:
            self.selected_square = None
        return None

    @staticmethod
    def _build_move(board: chess.Board, from_square: chess.Square, to_square: chess.Square) -> Optional[chess.Move]:
        """Build a move, auto-promoting to queen when a promotion is required."""
        piece = board.piece_at(from_square)
        promotion = None
        if piece is not None and piece.piece_type == chess.PAWN:
            target_rank = chess.square_rank(to_square)
            if target_rank in (0, 7):
                promotion = chess.QUEEN
        return chess.Move(from_square, to_square, promotion=promotion)

    def reset(self) -> None:
        """Clear the current selection."""
        self.selected_square = None
