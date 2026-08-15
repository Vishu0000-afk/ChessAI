"""Mid-level learning orchestration.

Owns the trainer, model, optimizer, replay buffer, and checkpoint manager.
The high-level coordinator drives workers and evaluation; the learner:

    train()          run TRAINING_STEPS mini-batch updates on the shared model
    checkpoint(...)  persist model + optimizer + metadata (bump version)
    resume()         reload the newest checkpoint after an interruption
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from src.learning.model_manager import CheckpointMetadata, ModelManager
from src.learning.network import create_chess_net
from src.learning.replay_buffer import ReplayBuffer
from src.learning.trainer import Trainer

logger = logging.getLogger(__name__)


class Learner:
    """Continuous-learning driver for a single shared model."""

    def __init__(
        self,
        model: nn.Module,
        replay_buffer: ReplayBuffer,
        model_manager: ModelManager,
        device: str = "cpu",
        batch_size: int = 512,
        training_steps: int = 100,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        use_mixed_precision: bool = True,
        mirror_augmentation: bool = True,
        compute_lock: Optional[threading.Lock] = None,
    ) -> None:
        self.model = model
        self.replay_buffer = replay_buffer
        self.model_manager = model_manager
        self.device = torch.device(device)
        self.training_steps_per_cycle = training_steps

        self.trainer = Trainer(
            model=model,
            device=device,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            use_mixed_precision=use_mixed_precision,
            mirror_augmentation=mirror_augmentation,
            compute_lock=compute_lock,
        )

        self.version = 0
        self.games_trained = 0
        self.total_training_steps = 0
        self.history: dict = {"policy_loss": [], "value_loss": [], "total_loss": []}
        self.last_train_summary: Optional[dict] = None

    # ------------------------------------------------------------------
    def resume(self, games_override: Optional[int] = None) -> CheckpointMetadata:
        """Restore the newest checkpoint; returns metadata or None."""
        meta = self.model_manager.resume(self.model, self.trainer.optimizer)
        if meta is not None:
            self.version = meta.version
            self.games_trained = meta.games_trained if games_override is None else games_override
            self.total_training_steps = meta.training_steps
            self.history = dict(meta.history)
        return meta

    def reset_model(self, in_channels: int = 18, channels: int = 64, res_blocks: int = 2) -> None:
        """Freshly reinitialize the network (used when no checkpoint exists)."""
        self.model = create_chess_net(in_channels=in_channels, channels=channels, res_blocks=res_blocks)
        self.trainer.model = self.model
        self.trainer.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.trainer.optimizer.param_groups[0]["lr"],
            weight_decay=self.trainer.optimizer.param_groups[0]["weight_decay"],
        )
        self.trainer.optimizer_state = None
        self.trainer.model.to(self.device)

    # ------------------------------------------------------------------
    def train(self, steps: Optional[int] = None, rng: Optional[np.random.Generator] = None) -> dict:
        """Run a training cycle on the replay buffer; returns a summary.

        The summary records the parameter L2 norm before and after so it is
        easy to verify (and prove) the weights actually changed.
        """
        if len(self.replay_buffer) < 1:
            logger.info("Replay buffer empty; skipping training cycle.")
            return {}

        steps = steps or self.training_steps_per_cycle
        norm_before = self.trainer.parameter_norm()
        summary = self.trainer.train(self.replay_buffer, steps)
        norm_after = self.trainer.parameter_norm()

        self.total_training_steps += steps
        for key in ("policy_loss", "value_loss", "total_loss"):
            if key in summary:
                self.history[key].append(summary[key])

        summary["param_norm_before"] = norm_before
        summary["param_norm_after"] = norm_after
        summary["param_delta"] = abs(norm_after - norm_before)
        summary["steps"] = steps
        summary["training_steps_total"] = self.total_training_steps
        self.last_train_summary = summary

        logger.info(
            "Training cycle done: steps=%d policy=%.4f value=%.4f |p| %.4f -> %.4f (d=%.5f)",
            steps,
            summary.get("policy_loss", 0.0),
            summary.get("value_loss", 0.0),
            norm_before,
            norm_after,
            summary["param_delta"],
        )
        return summary

    # ------------------------------------------------------------------
    def checkpoint(
        self,
        games_trained: int,
        version: Optional[int] = None,
        config: Optional[dict] = None,
        replay_info: Optional[dict] = None,
    ) -> str:
        """Save the current model under the given (default: current) version."""
        version = self.version if version is None else version
        meta = CheckpointMetadata(
            version=version,
            games_trained=games_trained,
            training_steps=self.total_training_steps,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            config=config or {},
            replay_info=replay_info or {"samples": len(self.replay_buffer)},
            history={k: list(v) for k, v in self.history.items()},
        )
        return self.model_manager.save(self.model, self.trainer.optimizer, meta)