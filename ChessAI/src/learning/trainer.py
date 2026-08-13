"""Neural-network trainer.

Samples mini-batches from the replay buffer and updates the network with a
real loss -> backward -> optimizer step. The value head uses MSE against the
game result; the policy head uses cross-entropy against the selected move,
masked so illegal moves can never receive probability mass.

Mixed precision (float16 on CUDA / bfloat16 on CPU) is applied inside an
autocast context. An optional external lock serializes access to the shared
module so training never races with batched inference on the same model.
"""

from __future__ import annotations

import logging
import threading
from contextlib import nullcontext
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from src.learning.replay_buffer import ReplayBuffer

logger = logging.getLogger(__name__)

_NEG_INF = float("-inf")


class Trainer:
    """Mini-batch trainer for :class:`ChessNet`."""

    def __init__(
        self,
        model: nn.Module,
        device: str = "cpu",
        batch_size: int = 512,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        use_mixed_precision: bool = True,
        compute_lock: Optional[threading.Lock] = None,
    ) -> None:
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)
        self.batch_size = batch_size
        self.use_mixed_precision = use_mixed_precision
        self.compute_lock = compute_lock
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        self.training_steps = 0

    # ------------------------------------------------------------------
    def train_step(self, buffer: ReplayBuffer, rng: Optional[np.random.Generator] = None) -> dict:
        """One optimizer step on a random mini-batch from the replay buffer."""
        batch = buffer.sample(self.batch_size, rng=rng)
        inputs = torch.from_numpy(np.ascontiguousarray(batch["input"])).to(self.device)
        move_targets = torch.from_numpy(batch["move_index"].astype(np.int64)).to(self.device)
        legal_packed = torch.from_numpy(batch["legal_packed"].astype(np.int64)).to(self.device)
        value_targets = torch.from_numpy(batch["value"]).to(self.device).unsqueeze(1)

        lock = self.compute_lock or nullcontext()
        with lock:
            self.model.train()
            self.optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type=self.device.type,
                enabled=self.use_mixed_precision and self.device.type in ("cuda", "cpu"),
                dtype=torch.float16 if self.device.type == "cuda" else torch.bfloat16,
            ):
                logits, values = self.model(inputs)
                masked = _mask_logits(logits, legal_packed)
                policy_loss = torch.nn.functional.cross_entropy(masked, move_targets)
                value_loss = torch.nn.functional.mse_loss(values, value_targets)
                total_loss = policy_loss + value_loss

            total_loss.backward()
            self.optimizer.step()

        self.training_steps += 1
        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "total_loss": total_loss.item(),
        }

    def train(self, buffer: ReplayBuffer, steps: int) -> dict:
        """Run ``steps`` mini-batch training steps; returns the last losses."""
        last = {}
        for _ in range(steps):
            last = self.train_step(buffer)
        return last

    def parameter_norm(self) -> float:
        """L2 norm of all trainable parameters (used to prove weights change)."""
        total = 0.0
        for p in self.model.parameters():
            if p.requires_grad:
                total += float(p.detach().to("cpu").norm().item() ** 2)
        return float(np.sqrt(total))

    def load_optimizer_state(self, state_dict) -> None:
        self.optimizer.load_state_dict(state_dict)


def _mask_logits(logits: torch.Tensor, legal_packed: torch.Tensor) -> torch.Tensor:
    """Set logits of illegal moves to -inf using bit-packed legal masks."""
    B = logits.size(0)
    bits = torch.arange(64, device=logits.device, dtype=legal_packed.dtype)
    mask = (legal_packed.unsqueeze(-1) >> bits) & 1  # (B, 64, 64)
    mask = mask.view(B, 4096).bool()
    return torch.where(mask, logits, torch.full_like(logits, _NEG_INF))