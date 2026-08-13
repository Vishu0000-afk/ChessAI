"""Rendering logic for the chess board and pieces using PyGame.

Kept independent of search/engine logic: this module only knows how
to draw a python-chess board plus some highlight state onto a PyGame
surface.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import chess
import pygame

LIGHT_SQUARE_COLOR = (240, 217, 181)
DARK_SQUARE_COLOR = (181, 136, 99)
SELECTED_COLOR = (246, 246, 105)
LEGAL_MOVE_COLOR = (106, 168, 79)
LAST_MOVE_COLOR = (205, 210, 106)
CHECK_COLOR = (220, 80, 80)

_PIECE_UNICODE: Dict[Tuple[chess.PieceType, chess.Color], str] = {
    (chess.PAWN, chess.WHITE): "\u2659",
    (chess.KNIGHT, chess.WHITE): "\u2658",
    (chess.BISHOP, chess.WHITE): "\u2657",
    (chess.ROOK, chess.WHITE): "\u2656",
    (chess.QUEEN, chess.WHITE): "\u2655",
    (chess.KING, chess.WHITE): "\u2654",
    (chess.PAWN, chess.BLACK): "\u265F",
    (chess.KNIGHT, chess.BLACK): "\u265E",
    (chess.BISHOP, chess.BLACK): "\u265D",
    (chess.ROOK, chess.BLACK): "\u265C",
    (chess.QUEEN, chess.BLACK): "\u265B",
    (chess.KING, chess.BLACK): "\u265A",
}


class BoardRenderer:
    """Draws a chess board, pieces, and highlights onto a PyGame surface."""

    def __init__(self, square_size: int, flipped: bool = False) -> None:
        """Create a renderer.

        Args:
            square_size: Pixel size of one board square.
            flipped: If True, render from Black's point of view.
        """
        self.square_size = square_size
        self.flipped = flipped
        pygame.font.init()
        self._piece_font = pygame.font.SysFont("segoeuisymbol", int(square_size * 0.7))
        if self._piece_font is None:
            self._piece_font = pygame.font.Font(None, int(square_size * 0.7))
        self._label_font = pygame.font.SysFont("arial", 16)

    def square_to_pixel(self, square: chess.Square) -> Tuple[int, int]:
        """Convert a chess square to top-left pixel coordinates."""
        file_idx = chess.square_file(square)
        rank_idx = chess.square_rank(square)
        if self.flipped:
            col = 7 - file_idx
            row = rank_idx
        else:
            col = file_idx
            row = 7 - rank_idx
        return col * self.square_size, row * self.square_size

    def pixel_to_square(self, x: int, y: int) -> Optional[chess.Square]:
        """Convert pixel coordinates to a chess square, or None if out of bounds."""
        col = x // self.square_size
        row = y // self.square_size
        if not (0 <= col <= 7 and 0 <= row <= 7):
            return None
        if self.flipped:
            file_idx = 7 - col
            rank_idx = row
        else:
            file_idx = col
            rank_idx = 7 - row
        return chess.square(file_idx, rank_idx)

    def draw(
        self,
        surface: pygame.Surface,
        board: chess.Board,
        selected_square: Optional[chess.Square] = None,
        legal_targets: Optional[List[chess.Square]] = None,
        last_move: Optional[chess.Move] = None,
        anim_move: Optional[chess.Move] = None,
        anim_progress: float = 0.0,
    ) -> None:
        """Draw the full board state onto the given surface.

        Args:
            surface: PyGame surface to draw on.
            board: Current chess position.
            selected_square: Currently selected square, if any.
            legal_targets: Squares the selected piece can legally move to.
            last_move: The most recently played move, for highlighting.
            anim_move: A move currently being animated (piece slides along it).
            anim_progress: Animation progress in [0, 1]; 0 is the start square,
                1 is the destination.
        """
        legal_targets = legal_targets or []
        self._draw_squares(surface)
        self._highlight_last_move(surface, last_move)
        self._highlight_selected(surface, selected_square)
        self._highlight_check(surface, board)
        self._draw_pieces(surface, board, anim_move=anim_move, anim_progress=anim_progress)
        self._highlight_legal_targets(surface, board, legal_targets)

    def _draw_squares(self, surface: pygame.Surface) -> None:
        for rank in range(8):
            for file in range(8):
                square = chess.square(file, rank)
                x, y = self.square_to_pixel(square)
                color = LIGHT_SQUARE_COLOR if (file + rank) % 2 == 0 else DARK_SQUARE_COLOR
                pygame.draw.rect(surface, color, (x, y, self.square_size, self.square_size))

    def _highlight_selected(self, surface: pygame.Surface, selected_square: Optional[chess.Square]) -> None:
        if selected_square is None:
            return
        x, y = self.square_to_pixel(selected_square)
        highlight = pygame.Surface((self.square_size, self.square_size), pygame.SRCALPHA)
        highlight.fill((*SELECTED_COLOR, 160))
        surface.blit(highlight, (x, y))

    def _highlight_last_move(self, surface: pygame.Surface, last_move: Optional[chess.Move]) -> None:
        if last_move is None:
            return
        for square in (last_move.from_square, last_move.to_square):
            x, y = self.square_to_pixel(square)
            highlight = pygame.Surface((self.square_size, self.square_size), pygame.SRCALPHA)
            highlight.fill((*LAST_MOVE_COLOR, 120))
            surface.blit(highlight, (x, y))

    def _highlight_check(self, surface: pygame.Surface, board: chess.Board) -> None:
        if not board.is_check():
            return
        king_square = board.king(board.turn)
        if king_square is None:
            return
        x, y = self.square_to_pixel(king_square)
        highlight = pygame.Surface((self.square_size, self.square_size), pygame.SRCALPHA)
        highlight.fill((*CHECK_COLOR, 140))
        surface.blit(highlight, (x, y))

    def _highlight_legal_targets(
        self, surface: pygame.Surface, board: chess.Board, legal_targets: List[chess.Square]
    ) -> None:
        for square in legal_targets:
            x, y = self.square_to_pixel(square)
            center = (x + self.square_size // 2, y + self.square_size // 2)
            is_capture = board.piece_at(square) is not None
            radius = self.square_size // 3 if is_capture else self.square_size // 6
            dot_surface = pygame.Surface((self.square_size, self.square_size), pygame.SRCALPHA)
            pygame.draw.circle(
                dot_surface,
                (*LEGAL_MOVE_COLOR, 160),
                (self.square_size // 2, self.square_size // 2),
                radius,
                width=0 if not is_capture else 4,
            )
            surface.blit(dot_surface, (x, y))

    def _draw_pieces(
        self,
        surface: pygame.Surface,
        board: chess.Board,
        anim_move: Optional[chess.Move] = None,
        anim_progress: float = 0.0,
    ) -> None:
        anim_center: Optional[Tuple[int, int]] = None
        if anim_move is not None:
            from_x, from_y = self.square_to_pixel(anim_move.from_square)
            to_x, to_y = self.square_to_pixel(anim_move.to_square)
            progress = min(max(anim_progress, 0.0), 1.0)
            anim_center = (
                int(from_x + (to_x - from_x) * progress + self.square_size // 2),
                int(from_y + (to_y - from_y) * progress + self.square_size // 2),
            )

        for square, piece in board.piece_map().items():
            if anim_move is not None and square == anim_move.from_square:
                continue  # Drawn separately at its animated position below.
            x, y = self.square_to_pixel(square)
            self._draw_piece(surface, piece, (x + self.square_size // 2, y + self.square_size // 2))

        if anim_center is not None:
            moving_piece = board.piece_at(anim_move.from_square)
            if moving_piece is not None:
                self._draw_piece(surface, moving_piece, anim_center)

    def _draw_piece(self, surface: pygame.Surface, piece: chess.Piece, center: Tuple[int, int]) -> None:
        symbol = _PIECE_UNICODE[(piece.piece_type, piece.color)]
        text_color = (255, 255, 255) if piece.color == chess.WHITE else (30, 30, 30)
        outline_color = (30, 30, 30) if piece.color == chess.WHITE else (255, 255, 255)
        # Simple outline effect for legibility on both square colors.
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            outline = self._piece_font.render(symbol, True, outline_color)
            rect = outline.get_rect(center=(center[0] + dx, center[1] + dy))
            surface.blit(outline, rect)
        text_surface = self._piece_font.render(symbol, True, text_color)
        rect = text_surface.get_rect(center=center)
        surface.blit(text_surface, rect)
