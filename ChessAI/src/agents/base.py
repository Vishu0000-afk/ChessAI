"""Common agent interface.

All bots implement ``select_move(board)``. Agents that can evaluate many
positions at once (neural agents) additionally implement
``predict_batch(positions)`` so self-play can share GPU inference.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

import chess


class ChessAgent(ABC):
    """Decision-maker that picks a move for the side to move."""

    name: str = "agent"
    version: Optional[int] = None  # model version this agent is running (if any)

    def __init__(self, name: Optional[str] = None) -> None:
        if name is not None:
            self.name = name

    @abstractmethod
    def select_move(self, board: chess.Board) -> Optional[chess.Move]:
        """Return the move the agent wants to play, or None if the game is over."""
        raise NotImplementedError

    def predict_batch(self, positions: List[chess.Board]):
        """Optional: evaluate many positions at once.

        Returns a tuple ``(policy_logits, values)`` of numpy arrays, or
        ``None`` if the agent does not support batched inference.
        """
        return None