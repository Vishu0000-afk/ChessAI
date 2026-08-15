"""Neural network agent.

Selects moves from the policy head of a neural network, masked to legal
moves. Supports stochastic play (temperature sampling) for self-play
exploration and deterministic play (temperature 0 = argmax) for evaluation.

Batched inference is provided through the injected *predictor*; in
multi-process self-play this is an ``InferenceClient`` shared with the GPU
inference server, so many workers never keep their own CUDA model.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import chess
import numpy as np

from src.agents.base import ChessAgent
from src.learning.encoding import move_to_index, POLICY_SIZE

EPSILON = 1e-9


class NeuralAgent(ChessAgent):
    """Agent that picks moves from neural policy logits."""

    name = "neural"

    def __init__(
        self,
        predictor,
        temperature: float = 1.0,
        version: Optional[int] = None,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(name=name)
        self.predictor = predictor
        self.temperature = temperature
        self.version = version
        self.rng = np.random.default_rng()

    def predict_batch(self, positions: List[chess.Board]) -> Tuple[np.ndarray, np.ndarray]:
        """Batch-evaluate positions: returns (logits[N,4096], values[N])."""
        return self.predictor.predict_batch(positions)

    def select_move(self, board: chess.Board) -> Optional[chess.Move]:
        moves = list(board.legal_moves)
        if not moves:
            return None
        logits, _ = self.predictor.predict_batch([board])
        return self._select_from_logits(board, moves, logits[0])

    def _select_from_logits(self, board: chess.Board, moves: List[chess.Move], logits: np.ndarray) -> chess.Move:
        """Sample/argmax a legal move from masked policy logits."""
        move_logits = np.array(
            [logits[move_to_index(m, board.turn)] for m in moves], dtype=np.float64
        )
        probs = _softmax(move_logits / max(self.temperature, EPSILON))

        if self.temperature <= 0.0:
            return moves[int(np.argmax(probs))]

        choice = self.rng.choice(len(moves), p=probs)
        return moves[int(choice)]

    @staticmethod
    def mask_logits(logits: np.ndarray, legal: List[chess.Move], stm: int = chess.WHITE) -> np.ndarray:
        """Set policy logits of illegal moves to -inf (for training loss)."""
        masked = np.full(POLICY_SIZE, -1e9, dtype=np.float32)
        for m in legal:
            masked[move_to_index(m, stm)] = logits[move_to_index(m, stm)]
        return masked


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    e = np.exp(x)
    return e / e.sum()