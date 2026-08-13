"""Top-level PyGame application: Human vs AI chess.

Coordinates the Board, ChessEngine, BoardRenderer, and InputHandler.
Kept as thin as reasonably possible: game state lives in Board, move
selection lives in InputHandler, drawing lives in BoardRenderer, and
move search lives in ChessEngine.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import chess
import pygame

from src.engine.board import Board
from src.gui.board_renderer import BoardRenderer
from src.gui.input_handler import InputHandler

logger = logging.getLogger(__name__)

WINDOW_WIDTH = 8 * 80 + 2 * 26  # board + coordinate gutters
WINDOW_HEIGHT = WINDOW_WIDTH + 60  # board + gutters + sidebar
SIDEBAR_HEIGHT = 60
SQUARE_SIZE = 80
GUTTER = 26
FPS = 30
AI_MOVE_DELAY = 2.0  # Seconds to wait after the last move before the AI plays.
ANIM_DURATION = 0.5  # Seconds a piece takes to slide between squares.

BACKGROUND_COLOR = (228, 222, 208)  # light neutral beige
TEXT_COLOR = (45, 45, 45)
SIDEBAR_COLOR = (216, 210, 194)

MIN_SQUARE_SIZE = 24
MAX_SQUARE_SIZE = 100


class ChessGUI:
    """PyGame application running a Human vs AI chess game."""

    def __init__(
        self,
        ai_color: chess.Color = chess.BLACK,
        search_depth: int = 3,
        agent: Optional["ChessAgent"] = None,
    ) -> None:
        """Create the GUI application.

        Args:
            ai_color: Which color the engine plays (chess.WHITE or chess.BLACK).
            search_depth: Search depth used if no agent is supplied.
            agent: Optional bot to play as the AI. If None, a classical
                engine agent at ``search_depth`` is created.
        """
        pygame.init()
        pygame.display.set_caption("ChessAI")
        self.window_w = WINDOW_WIDTH
        self.window_h = WINDOW_HEIGHT
        self.screen = pygame.display.set_mode((self.window_w, self.window_h), pygame.RESIZABLE)
        self.clock = pygame.time.Clock()

        self.ai_color = ai_color
        self.human_color = not ai_color
        self.renderer = BoardRenderer(
            SQUARE_SIZE, flipped=(self.human_color == chess.BLACK), gutter=GUTTER
        )
        self.input_handler = InputHandler(self.renderer)
        if agent is None:
            from src.agents.classical_engine import ClassicalEngineAgent

            agent = ClassicalEngineAgent(depth=search_depth)
        self.agent = agent

        self.board = Board.new_game()
        self.last_move: Optional[chess.Move] = None
        self.running = True
        self.status_message = ""
        self._last_move_time = time.monotonic()
        self._anim_move: Optional[chess.Move] = None
        self._anim_start = 0.0

        self._font = pygame.font.SysFont("arial", 22)

    def run(self) -> None:
        """Run the main event loop until the window is closed."""
        logger.info("Starting ChessAI GUI. Human=%s AI=%s", self.human_color, self.ai_color)
        while self.running:
            self._handle_events()

            if self._anim_move is not None:
                if time.monotonic() - self._anim_start >= ANIM_DURATION:
                    self._apply_animated_move()
            elif (
                not self.board.is_game_over()
                and self.board.turn() == self.ai_color
                and time.monotonic() - self._last_move_time >= AI_MOVE_DELAY
            ):
                self._make_ai_move()

            self._draw()
            self.clock.tick(FPS)

        pygame.quit()

    def _handle_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    self._restart()
            elif event.type == pygame.VIDEORESIZE:
                self._apply_resize(event.w, event.h)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                self._handle_click(event.pos)

    def _apply_resize(self, width: int, height: int) -> None:
        """Recompute the square size/origin and rebuild the renderer layout."""
        self.window_w, self.window_h = width, height
        avail_w = width - 2 * GUTTER
        avail_h = height - SIDEBAR_HEIGHT - 2 * GUTTER
        board_px = min(avail_w, avail_h, 8 * MAX_SQUARE_SIZE)
        board_px = max(board_px, 8 * MIN_SQUARE_SIZE)
        sq = board_px // 8
        board_px = sq * 8
        ox = (width - board_px) // 2
        oy = (height - SIDEBAR_HEIGHT - board_px) // 2
        self.renderer.resize(sq, ox, oy)
        self.screen = pygame.display.set_mode((width, height), pygame.RESIZABLE)

    def _handle_click(self, pos: tuple) -> None:
        if self._anim_move is not None:
            return
        if self.board.is_game_over():
            return
        if self.board.turn() != self.human_color:
            return

        x, y = pos
        if y >= self.window_h - SIDEBAR_HEIGHT:
            return  # Click landed in the sidebar area.

        move = self.input_handler.handle_click(self.board.raw, x, y)
        if move is not None:
            if not self.board.is_legal(move):
                logger.warning("Rejected illegal move attempt: %s", move)
                return
            self._last_move_time = time.monotonic()
            self._start_animation(move)
            logger.info("Human played %s", move.uci())

    def _make_ai_move(self) -> None:
        move = self.agent.select_move(self.board.raw)
        if move is None:
            logger.info("Engine found no move; game must be over.")
            return
        stats = self.engine_stats()
        logger.info(
            "AI chose %s | depth=%d nodes=%d time=%.2fs nps=%.0f eval=%d",
            move.uci(),
            stats.depth,
            stats.nodes_searched,
            stats.search_time_seconds,
            stats.nps,
            stats.evaluation,
        )
        self._start_animation(move)

    def engine_stats(self):
        """Best-effort stats snapshot from the underlying classical engine."""
        engine = getattr(self.agent, "engine", None)
        if engine is not None:
            return engine.get_stats()
        return self._empty_stats()

    def _empty_stats(self):
        from src.engine.engine import EngineStats

        return EngineStats()

    def _start_animation(self, move: chess.Move) -> None:
        self._anim_move = move
        self._anim_start = time.monotonic()

    def _apply_animated_move(self) -> None:
        if self._anim_move is None:
            return
        was_ai_turn = self.board.turn() == self.ai_color
        move = self._anim_move
        self.board.make_move(move)
        self.last_move = move
        self._anim_move = None
        self._last_move_time = time.monotonic()
        if was_ai_turn:
            logger.info("AI played %s", move.uci())

    def _restart(self) -> None:
        logger.info("Restarting game.")
        self.board = Board.new_game()
        self.last_move = None
        self._anim_move = None
        self.input_handler.reset()
        reset = getattr(self.agent, "reset", None)
        if callable(reset):
            reset()

    def _draw(self) -> None:
        self.screen.fill(BACKGROUND_COLOR)

        selected = self.input_handler.selected_square
        targets = self.input_handler.legal_targets(self.board.raw) if selected is not None else []
        anim_progress = 0.0
        if self._anim_move is not None:
            anim_progress = (time.monotonic() - self._anim_start) / ANIM_DURATION
        self.renderer.draw(
            self.screen,
            self.board.raw,
            selected_square=selected,
            legal_targets=targets,
            last_move=self.last_move,
            anim_move=self._anim_move,
            anim_progress=anim_progress,
        )
        self._draw_sidebar()
        pygame.display.flip()

    def _draw_sidebar(self) -> None:
        y_offset = self.window_h - SIDEBAR_HEIGHT
        pygame.draw.rect(self.screen, SIDEBAR_COLOR, (0, y_offset, self.window_w, SIDEBAR_HEIGHT))

        status = self._status_text()
        text_surface = self._font.render(status, True, TEXT_COLOR)
        self.screen.blit(text_surface, (10, y_offset + 18))

    def _status_text(self) -> str:
        if self.board.is_checkmate():
            winner = "White" if self.board.turn() == chess.BLACK else "Black"
            return f"Checkmate — {winner} wins! Press R to restart."
        if self.board.is_stalemate():
            return "Stalemate — draw. Press R to restart."
        if self.board.is_draw():
            return "Draw. Press R to restart."
        turn_name = "White" if self.board.turn() == chess.WHITE else "Black"
        check_suffix = " (check)" if self.board.is_check() else ""
        return f"{turn_name} to move{check_suffix}   [R: restart]"
