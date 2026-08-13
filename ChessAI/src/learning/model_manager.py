"""Model checkpointing, versioning, and resume-after-crash support.

Checkpoints are ``torch.save`` files containing model weights, optimizer
state, and metadata (version, games trained, training steps, config snapshot,
replay-buffer info, timestamp). A monotonically increasing ``version`` is
assigned to every saved model; ``latest.pth`` always mirrors the newest
checkpoint so training can resume transparently.
"""

from __future__ import annotations

import glob
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

_VERSION_PATTERN = re.compile(r".*_(\d{6})\.pth$")


@dataclass
class CheckpointMetadata:
    """Snapshot describing a saved model."""

    version: int
    games_trained: int
    training_steps: int
    timestamp: str
    config: Dict[str, object] = field(default_factory=dict)
    replay_info: Dict[str, object] = field(default_factory=dict)
    history: Dict[str, object] = field(default_factory=dict)


class ModelManager:
    """Owns the checkpoint directory and model version numbering."""

    def __init__(
        self,
        checkpoint_dir: str,
        save_name_template: str = "model_{version:06d}.pth",
        latest_name: str = "latest.pth",
    ) -> None:
        self.checkpoint_dir = checkpoint_dir
        self.save_name_template = save_name_template
        self.latest_name = latest_name
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    # ------------------------------------------------------------------
    def save(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        metadata: CheckpointMetadata,
    ) -> str:
        """Save a checkpoint; returns the written path."""
        versioned_path = os.path.join(
            self.checkpoint_dir, self.save_name_template.format(version=metadata.version)
        )
        latest_path = os.path.join(self.checkpoint_dir, self.latest_name)

        payload = {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
            "metadata": {
                "version": metadata.version,
                "games_trained": metadata.games_trained,
                "training_steps": metadata.training_steps,
                "timestamp": metadata.timestamp,
                "config": metadata.config,
                "replay_info": metadata.replay_info,
                "history": metadata.history,
            },
        }
        torch.save(payload, versioned_path)
        torch.save(payload, latest_path)
        logger.info("Saved checkpoint version=%d games=%d -> %s", metadata.version, metadata.games_trained, versioned_path)
        return versioned_path

    # ------------------------------------------------------------------
    def _try_load(self, path: str):
        try:
            return torch.load(path, map_location="cpu", weights_only=False)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Failed to load checkpoint %s: %s", path, exc)
            return None

    def find_latest(self) -> Optional[str]:
        """Path of the newest numbered checkpoint (or latest.pth)."""
        numbered = sorted(glob.glob(os.path.join(self.checkpoint_dir, "model_*.pth")))
        if numbered:
            return numbered[-1]
        latest = os.path.join(self.checkpoint_dir, self.latest_name)
        return latest if os.path.exists(latest) else None

    def list_checkpoints(self) -> List[Tuple[int, str]]:
        """All (version, path) pairs found in the checkpoint directory."""
        out = []
        for path in glob.glob(os.path.join(self.checkpoint_dir, "model_*.pth")):
            m = _VERSION_PATTERN.match(os.path.basename(path))
            if m:
                out.append((int(m.group(1)), path))
        return sorted(out)

    def next_version(self) -> int:
        versions = [v for v, _ in self.list_checkpoints()]
        return max(versions) + 1 if versions else 1

    def resume(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> Optional[CheckpointMetadata]:
        """Restore the newest checkpoint onto ``model`` (and optimizer)."""
        path = self.find_latest()
        if path is None:
            return None
        payload = self._try_load(path)
        if payload is None:
            return None
        model.load_state_dict(payload["model_state"])
        if optimizer is not None and payload.get("optimizer_state") is not None:
            optimizer.load_state_dict(payload["optimizer_state"])
        meta = CheckpointMetadata(**payload["metadata"])
        logger.info("Resumed from %s (version=%d games=%d)", path, meta.version, meta.games_trained)
        return meta