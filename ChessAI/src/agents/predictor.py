"""Board predictors used by neural agents.

A predictor turns a list of ``chess.Board`` positions into policy logits and
value estimates. Two implementations exist:

* :class:`LocalPredictor` — holds a torch model directly (single-process use:
  GUI, evaluator, tests).
* ``InferenceClient`` (see ``src/selfplay/inference.py``) — talks to a shared
  batched-inference server through queues (multi-process self-play), so one
  GPU model is shared by many CPU workers.

Both conform to the same ``predict_batch`` protocol.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import chess
import numpy as np
import torch

from src.learning.encoding import encode_board_batch

Logits = np.ndarray  # (N, 4096)
Values = np.ndarray  # (N,)


class LocalPredictor:
    """Direct in-process batched predictor around a torch model."""

    def __init__(self, model: torch.nn.Module, device: str = "cpu", name: str = "local") -> None:
        self.model = model
        self.device = device
        self.name = name
        self._lock = None  # optional external lock passed by coordinator

    def predict_batch(self, positions: List[chess.Board]) -> Tuple[Logits, Values]:
        if not positions:
            return np.zeros((0, 4096), dtype=np.float32), np.zeros(0, dtype=np.float32)
        x = torch.from_numpy(encode_board_batch(positions)).to(self.device)
        with torch.no_grad():
            logits, value = self.model(x)
        return logits.detach().cpu().numpy(), value.detach().cpu().numpy().reshape(-1)