"""Rendering logic for the chess board and pieces using PyGame.

Kept independent of search/engine logic: this module only knows how
to draw a python-chess board plus some highlight state onto a PyGame
surface.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import chess
import pygame

# Flat classic palette: deep forest green + warm cream, thin boundaries.
LIGHT_SQUARE_COLOR = (241, 231, 204)  # warm ivory/cream
DARK_SQUARE_COLOR = (73, 110, 84)  # deep muted forest green
GRID_COLOR = (56, 88, 68)  # subtle 1px boundary between squares
BORDER_COLOR = (168, 162, 148)  # subtle neutral beige outer frame
COORD_COLOR = (45, 50, 44)  # dark gray coordinate labels
SELECTED_COLOR = (255, 210, 92)  # warm gold
LEGAL_MOVE_COLOR = (255, 255, 255)  # soft white dots/rings
LAST_MOVE_COLOR = (232, 192, 66)  # muted yellow
CHECK_COLOR = (232, 64, 64)

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

    def __init__(self, square_size: int, flipped: bool = False, gutter: int = 0) -> None:
        """Create a renderer.

        Args:
            square_size: Pixel size of one board square.
            flipped: If True, render from Black's point of view.
            gutter: Margin reserved around the board for coordinates.
        """
        self.square_size = square_size
        self.flipped = flipped
        self.gutter = gutter
        self.origin_x = gutter
        self.origin_y = gutter
        pygame.font.init()
        self._init_fonts()

    def _init_fonts(self) -> None:
        self._piece_font = pygame.font.SysFont("segoeuisymbol", int(self.square_size * 0.72))
        if self._piece_font is None:
            self._piece_font = pygame.font.Font(None, int(self.square_size * 0.72))
        self._label_font = pygame.font.SysFont("arial", 15)

    def resize(self, square_size: int, origin_x: int, origin_y: int) -> None:
        """Update layout after a window resize; keeps the same orientation."""
        self.square_size = square_size
        self.origin_x = origin_x
        self.origin_y = origin_y
        self._init_fonts()

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
        return self.origin_x + col * self.square_size, self.origin_y + row * self.square_size

    def pixel_to_square(self, x: int, y: int) -> Optional[chess.Square]:
        """Convert pixel coordinates to a chess square, or None if out of bounds."""
        col = (x - self.origin_x) // self.square_size
        row = (y - self.origin_y) // self.square_size
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
        self._draw_coordinates(surface)
        self._highlight_legal_targets(surface, board, legal_targets)

    def _draw_squares(self, surface: pygame.Surface) -> None:
        sq = self.square_size
        ox, oy = self.origin_x, self.origin_y
        board_px = sq * 8
        # Thin neutral outer frame.
        pygame.draw.rect(surface, BORDER_COLOR, (ox - 2, oy - 2, board_px + 4, board_px + 4))
        for rank in range(8):
            for file in range(8):
                square = chess.square(file, rank)
                x, y = self.square_to_pixel(square)
                color = LIGHT_SQUARE_COLOR if (file + rank) % 2 == 0 else DARK_SQUARE_COLOR
                pygame.draw.rect(surface, color, (x, y, sq, sq))
        # Subtle 1px boundaries between squares.
        for i in range(1, 8):
            pygame.draw.line(surface, GRID_COLOR, (ox + i * sq, oy), (ox + i * sq, oy + board_px))
            pygame.draw.line(surface, GRID_COLOR, (ox, oy + i * sq), (ox + board_px, oy + i * sq))

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

    def _draw_coordinates(self, surface: pygame.Surface) -> None:
        """Draw file letters (a-h) centered above/below each column and rank
        numbers (1-8) centered beside each row, in the margin around the
        board. Labels track the current orientation automatically."""
        sq = self.square_size
        ox, oy = self.origin_x, self.origin_y
        board_px = sq * 8

        for col in range(8):
            bx = ox + col * sq + sq // 2
            top_square = self.pixel_to_square(bx, oy)
            bottom_square = self.pixel_to_square(bx, oy + board_px - 1)
            if top_square is None:
                continue
            file_name = chess.FILE_NAMES[chess.square_file(top_square)]
            label = self._label_font.render(file_name, True, COORD_COLOR)
            # Top and bottom edges.
            surface.blit(label, (bx - label.get_width() // 2, oy - label.get_height() - 4))
            if bottom_square is not None:
                surface.blit(label, (bx - label.get_width() // 2, oy + board_px + 3))

        for row in range(8):
            by = oy + row * sq + sq // 2
            left_square = self.pixel_to_square(ox + 1, by)
            right_square = self.pixel_to_square(ox + board_px - 1, by)
            if left_square is None:
                continue
            rank_name = chess.RANK_NAMES[chess.square_rank(left_square)]
            label = self._label_font.render(rank_name, True, COORD_COLOR)
            # Left and right edges.
            surface.blit(label, (ox - label.get_width() - 5, by - label.get_height() // 2))
            if right_square is not None:
                surface.blit(label, (ox + board_px + 5, by - label.get_height() // 2))

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
