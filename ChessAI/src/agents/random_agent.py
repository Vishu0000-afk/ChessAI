"""Uniform-random agent.

Plays a uniformly random legal move. Useful as a trivial baseline for
evaluation sanity checks and tests (never legal-problematic).
"""

from __future__ import annotations

import random
from typing import Optional

import chess

from src.agents.base import ChessAgent


class RandomAgent(ChessAgent):
    """Picks a uniformly random legal move."""

    name = "random"

    def __init__(self, rng: Optional[random.Random] = None, name: Optional[str] = None) -> None:
        super().__init__(name=name)
        self.rng = rng or random.Random()

    def select_move(self, board: chess.Board) -> Optional[chess.Move]:
        moves = list(board.legal_moves)
        if not moves:
            return None
        return self.rng.choice(moves)