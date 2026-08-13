"""High-level ChessEngine coordinating board, evaluation, and search."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import chess

from src.engine.board import Board
from src.engine.evaluator import Evaluator
from src.engine.search import Search, SearchResult
from src.engine.transposition import TranspositionTable

logger = logging.getLogger(__name__)


@dataclass
class EngineStats:
    """Snapshot of statistics from the engine's most recent search."""

    depth: int = 0
    nodes_searched: int = 0
    evaluation: int = 0
    search_time_seconds: float = 0.0
    nps: float = 0.0
    tt_hits: int = 0
    tt_size: int = 0


class ChessEngine:
    """Coordinates the board, evaluator, search, and transposition table.

    Example:
        >>> engine = ChessEngine(depth=4)
        >>> board = Board.new_game()
        >>> move = engine.get_best_move(board)
    """

    def __init__(
        self,
        depth: int = 3,
        use_transposition_table: bool = True,
        use_move_ordering: bool = True,
    ) -> None:
        """Create a chess engine.

        Args:
            depth: Default search depth in plies.
            use_transposition_table: Whether to cache search results.
            use_move_ordering: Whether to order moves for better pruning.
        """
        if depth < 1:
            raise ValueError("depth must be at least 1.")

        self.depth = depth
        self.evaluator = Evaluator()
        self.transposition_table = TranspositionTable(enabled=use_transposition_table)
        self.search_engine = Search(
            evaluator=self.evaluator,
            transposition_table=self.transposition_table,
            use_move_ordering=use_move_ordering,
        )
        self._last_result: Optional[SearchResult] = None

    def get_best_move(self, board: Board, depth: Optional[int] = None) -> Optional[chess.Move]:
        """Search for and return the best move in the given position.

        Args:
            board: Current game position.
            depth: Optional override for search depth (defaults to
                the engine's configured depth).

        Returns:
            The best move found, or None if the game is already over.
        """
        search_depth = depth if depth is not None else self.depth
        logger.info("Engine searching: depth=%d fen=%s", search_depth, board.get_fen())

        result = self.search_engine.search(board.raw, search_depth)
        self._last_result = result

        if result.best_move is None:
            logger.info("No legal moves available; game is over.")
        return result.best_move

    def evaluate_position(self, board: Board) -> int:
        """Return the static evaluation of a position (White's perspective)."""
        return self.evaluator.evaluate(board.raw)

    def get_stats(self) -> EngineStats:
        """Return statistics from the most recent search.

        Returns:
            An EngineStats snapshot. Fields are zeroed if no search
            has been performed yet.
        """
        if self._last_result is None:
            return EngineStats(tt_size=len(self.transposition_table))
        r = self._last_result
        return EngineStats(
            depth=r.depth,
            nodes_searched=r.nodes_searched,
            evaluation=r.score,
            search_time_seconds=r.time_seconds,
            nps=r.nps,
            tt_hits=self.transposition_table.hits,
            tt_size=len(self.transposition_table),
        )

    def reset(self) -> None:
        """Clear the transposition table and search history."""
        self.transposition_table.clear()
        self._last_result = None
