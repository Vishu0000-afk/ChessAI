"""Classical alpha-beta engine agent.

Wraps the existing ``ChessEngine`` (material + PST + negamax search) so it
can be used as a self-play / evaluation baseline bot::

    Bot A = NeuralAgent
    Bot B = ClassicalEngineAgent
"""

from __future__ import annotations

from typing import Optional

import chess

from src.agents.base import ChessAgent
from src.engine.board import Board
from src.engine.engine import ChessEngine


class ClassicalEngineAgent(ChessAgent):
    """Agent backed by the classical minimax engine."""

    name = "classical-engine"

    def __init__(self, depth: int = 3, name: Optional[str] = None) -> None:
        super().__init__(name=name)
        self.depth = depth
        self.engine = ChessEngine(depth=depth)

    def select_move(self, board: chess.Board) -> Optional[chess.Move]:
        return self.engine.get_best_move(Board.from_raw(board))

    def reset(self) -> None:
        """Clear the engine's transposition table between games."""
        self.engine.reset()