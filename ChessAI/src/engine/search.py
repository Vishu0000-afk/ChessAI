"""Minimax search with alpha-beta pruning.

Implemented as negamax (a common simplification of minimax that
exploits the zero-sum symmetry of chess) with alpha-beta pruning,
optional move ordering, and an optional transposition table.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional

import chess

from src.engine.evaluator import MATE_SCORE, Evaluator
from src.engine.move_ordering import order_moves
from src.engine.transposition import Bound, TranspositionTable

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Result of a search call."""

    best_move: Optional[chess.Move]
    score: int
    depth: int
    nodes_searched: int = 0
    time_seconds: float = 0.0

    @property
    def nps(self) -> float:
        """Nodes searched per second."""
        if self.time_seconds <= 0:
            return 0.0
        return self.nodes_searched / self.time_seconds


@dataclass
class SearchStats:
    """Mutable counters accumulated during a single search call."""

    nodes_searched: int = 0
    tt_hits: int = 0


class Search:
    """Alpha-beta search over a chess position."""

    def __init__(
        self,
        evaluator: Evaluator,
        transposition_table: Optional[TranspositionTable] = None,
        use_move_ordering: bool = True,
    ) -> None:
        """Create a search instance.

        Args:
            evaluator: Evaluator used to score leaf/terminal positions.
            transposition_table: Optional table for caching search results.
                If None, a disabled table is used internally (search still
                works correctly, just without caching).
            use_move_ordering: Whether to order moves before searching them.
        """
        self.evaluator = evaluator
        self.tt = transposition_table or TranspositionTable(enabled=False)
        self.use_move_ordering = use_move_ordering

    def search(self, board: chess.Board, depth: int) -> SearchResult:
        """Search the position to the given depth and return the best move.

        Args:
            board: Position to search from (side to move is respected).
            depth: Number of plies to search.

        Returns:
            A SearchResult containing the best move, its score (from the
            perspective of the side to move, in centipawns), node count,
            and elapsed time.

        Raises:
            ValueError: If depth is less than 1.
        """
        if depth < 1:
            raise ValueError("Search depth must be at least 1.")

        start_time = time.perf_counter()
        stats = SearchStats()

        legal_moves = list(board.legal_moves)
        if not legal_moves:
            # No legal moves: checkmate or stalemate at the root.
            score = self._terminal_score(board, ply=0)
            elapsed = time.perf_counter() - start_time
            logger.info("Search at root found no legal moves (game over).")
            return SearchResult(
                best_move=None, score=score, depth=depth, nodes_searched=0, time_seconds=elapsed
            )

        alpha = -MATE_SCORE - 1
        beta = MATE_SCORE + 1
        best_move: Optional[chess.Move] = None
        best_score = -MATE_SCORE - 1

        tt_move = self._probe_tt_move(board)
        if self.use_move_ordering:
            legal_moves = order_moves(board, legal_moves, tt_move=tt_move)

        for move in legal_moves:
            board.push(move)
            score = -self._negamax(board, depth - 1, -beta, -alpha, stats, ply=1)
            board.pop()

            if score > best_score:
                best_score = score
                best_move = move
            if best_score > alpha:
                alpha = best_score

        elapsed = time.perf_counter() - start_time
        logger.info(
            "Search complete: depth=%d nodes=%d time=%.3fs best_move=%s score=%d tt_hits=%d",
            depth,
            stats.nodes_searched,
            elapsed,
            best_move.uci() if best_move else None,
            best_score,
            stats.tt_hits,
        )

        return SearchResult(
            best_move=best_move,
            score=best_score,
            depth=depth,
            nodes_searched=stats.nodes_searched,
            time_seconds=elapsed,
        )

    def _negamax(
        self,
        board: chess.Board,
        depth: int,
        alpha: int,
        beta: int,
        stats: SearchStats,
        ply: int,
    ) -> int:
        """Negamax search with alpha-beta pruning.

        Returns the score from the perspective of the side to move at
        this node (higher is always better for whoever is to move).
        """
        stats.nodes_searched += 1
        original_alpha = alpha

        tt_key = self.tt.compute_key(board) if self.tt.enabled else None
        tt_move: Optional[chess.Move] = None
        if tt_key is not None:
            entry = self.tt.lookup(tt_key, depth, alpha, beta)
            if entry is not None:
                tt_move = entry.best_move
                if entry.bound == Bound.EXACT:
                    return entry.score
                if entry.bound == Bound.LOWER:
                    alpha = max(alpha, entry.score)
                elif entry.bound == Bound.UPPER:
                    beta = min(beta, entry.score)
                if alpha >= beta:
                    return entry.score

        if board.is_checkmate():
            # The side to move is checkmated: worst possible score,
            # adjusted by ply so faster mates are preferred/avoided correctly.
            return -(MATE_SCORE - ply)
        if board.is_stalemate() or board.is_insufficient_material():
            return 0
        if depth == 0:
            return self._relative_eval(board)

        legal_moves = list(board.legal_moves)
        if not legal_moves:
            # Should be covered by checkmate/stalemate checks above, but
            # guard defensively.
            return self._terminal_score(board, ply)

        if self.use_move_ordering:
            legal_moves = order_moves(board, legal_moves, tt_move=tt_move)

        best_score = -MATE_SCORE - 1
        best_move: Optional[chess.Move] = None

        for move in legal_moves:
            board.push(move)
            score = -self._negamax(board, depth - 1, -beta, -alpha, stats, ply + 1)
            board.pop()

            if score > best_score:
                best_score = score
                best_move = move
            if best_score > alpha:
                alpha = best_score
            if alpha >= beta:
                break  # Beta cutoff.

        if tt_key is not None:
            if best_score <= original_alpha:
                bound = Bound.UPPER
            elif best_score >= beta:
                bound = Bound.LOWER
            else:
                bound = Bound.EXACT
            self.tt.store(tt_key, depth, best_score, best_move, bound)

        return best_score

    def _relative_eval(self, board: chess.Board) -> int:
        """Evaluate the board from the perspective of the side to move."""
        white_perspective_score = self.evaluator.evaluate(board)
        return white_perspective_score if board.turn == chess.WHITE else -white_perspective_score

    def _terminal_score(self, board: chess.Board, ply: int) -> int:
        """Score a position with no legal moves (checkmate or stalemate)."""
        if board.is_checkmate():
            return -(MATE_SCORE - ply)
        return 0

    def _probe_tt_move(self, board: chess.Board) -> Optional[chess.Move]:
        """Best-effort lookup of a stored best move for ordering purposes."""
        if not self.tt.enabled:
            return None
        key = self.tt.compute_key(board)
        entry = self.tt.lookup(key, depth=0, alpha=-MATE_SCORE, beta=MATE_SCORE)
        return entry.best_move if entry else None
