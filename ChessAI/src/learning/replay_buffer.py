"""Ring-buffer replay store for self-play experiences.

Each experience is:

    input        float32 (18, 8, 8)   encoded position
    move_index   int16                policy target (selected move index)
    legal_packed uint64 (64,)         bit-packed legal-move mask (4096 bits)
    value        float32              result from the side-to-move's view
    version      int32                model version that produced the sample

The buffer is a pre-allocated numpy ring (no per-sample Python objects),
samples are returned as stacked arrays, and it supports persistent
save/load so training can resume after an interruption.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from src.learning.encoding import NUM_CHANNELS, POLICY_SIZE

_PACKED_WORDS = POLICY_SIZE // 64  # 64


class ReplayBuffer:
    """Fixed-capacity ring buffer of training experiences."""

    def __init__(self, capacity: int) -> None:
        if capacity < 1:
            raise ValueError("Replay buffer capacity must be >= 1.")
        self.capacity = int(capacity)
        self.inputs = np.zeros((self.capacity, NUM_CHANNELS, 8, 8), dtype=np.float32)
        self.move_indices = np.zeros(self.capacity, dtype=np.int16)
        self.legal_packed = np.zeros((self.capacity, _PACKED_WORDS), dtype=np.uint64)
        self.values = np.zeros(self.capacity, dtype=np.float32)
        self.versions = np.zeros(self.capacity, dtype=np.int32)
        self.size = 0
        self.head = 0  # next write slot

    # ------------------------------------------------------------------
    def extend(self, samples: List[Dict]) -> None:
        """Append a list of experience dicts, evicting the oldest as needed."""
        if not samples:
            return
        inputs = np.stack([s["input"] for s in samples]).astype(np.float32)
        move_idx = np.array([s["move_index"] for s in samples], dtype=np.int16)
        legal = np.stack([s["legal_packed"] for s in samples]).astype(np.uint64)
        values = np.array([s["value"] for s in samples], dtype=np.float32)
        versions = np.array([s["version"] for s in samples], dtype=np.int32)
        self._add_arrays(inputs, move_idx, legal, values, versions)

    def _add_arrays(self, inputs, move_idx, legal, values, versions) -> None:
        n = len(inputs)
        if n >= self.capacity:
            # Keep only the newest `capacity` samples.
            inputs = inputs[-self.capacity:]
            move_idx = move_idx[-self.capacity:]
            legal = legal[-self.capacity:]
            values = values[-self.capacity:]
            versions = versions[-self.capacity:]
            n = self.capacity

        end = self.head + n
        if end <= self.capacity:
            self.inputs[self.head:end] = inputs
            self.move_indices[self.head:end] = move_idx
            self.legal_packed[self.head:end] = legal
            self.values[self.head:end] = values
            self.versions[self.head:end] = versions
        else:
            first = self.capacity - self.head
            self.inputs[self.head:] = inputs[:first]
            self.move_indices[self.head:] = move_idx[:first]
            self.legal_packed[self.head:] = legal[:first]
            self.values[self.head:] = values[:first]
            self.versions[self.head:] = versions[:first]

            self.inputs[:n - first] = inputs[first:]
            self.move_indices[:n - first] = move_idx[first:]
            self.legal_packed[:n - first] = legal[first:]
            self.values[:n - first] = values[first:]
            self.versions[:n - first] = versions[first:]

        self.head = (self.head + n) % self.capacity
        self.size = min(self.capacity, self.size + n)

    def add(self, sample: Dict) -> None:
        self.extend([sample])

    # ------------------------------------------------------------------
    def sample(self, batch_size: int, rng: Optional[np.random.Generator] = None) -> Dict[str, np.ndarray]:
        """Uniformly sample a mini-batch of stacked arrays."""
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self.size == 0:
            raise ValueError("Cannot sample from an empty replay buffer.")
        rng = rng or np.random.default_rng()
        idx = rng.integers(0, self.size, size=batch_size)
        return self._gather(idx)

    def _valid_indices(self) -> np.ndarray:
        return (self.head - self.size + np.arange(self.size)) % self.capacity

    def _gather(self, idx: np.ndarray) -> Dict[str, np.ndarray]:
        return {
            "input": self.inputs[idx],
            "move_index": self.move_indices[idx],
            "legal_packed": self.legal_packed[idx],
            "value": self.values[idx],
            "version": self.versions[idx],
        }

    # ------------------------------------------------------------------
    def to_numpy(self) -> Dict[str, np.ndarray]:
        """Return all stored samples as contiguous stacked arrays."""
        return self._gather(self._valid_indices())

    def save(self, path: str) -> None:
        """Persist the buffer to a compressed .npz file."""
        data = self.to_numpy()
        np.savez_compressed(
            path,
            input=data["input"],
            move_index=data["move_index"],
            legal_packed=data["legal_packed"],
            value=data["value"],
            version=data["version"],
        )

    def load(self, path: str) -> None:
        """Restore the buffer from a .npz file produced by :meth:`save`."""
        with np.load(path) as z:
            inputs = z["input"]
            samples = []
            for i in range(len(inputs)):
                samples.append(
                    {
                        "input": inputs[i],
                        "move_index": z["move_index"][i],
                        "legal_packed": z["legal_packed"][i],
                        "value": z["value"][i],
                        "version": z["version"][i],
                    }
                )
        self.clear()
        self.extend(samples)

    def clear(self) -> None:
        self.size = 0
        self.head = 0

    def __len__(self) -> int:
        return self.size

    @property
    def is_full(self) -> bool:
        return self.size >= self.capacity