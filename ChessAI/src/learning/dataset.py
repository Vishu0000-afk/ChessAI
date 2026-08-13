"""Chunked serialized storage for self-play training data.

Experiences are stored as compressed numpy chunks (``chunk_XXXXXX.npz``)
under ``data/self_play/`` instead of millions of Python objects in RAM.
The chunk format mirrors the replay buffer layout (input / move_index /
legal_packed / value / version).

A torch ``Dataset`` is provided so future pipelines can train straight from
chunks (e.g. for very large offline training runs).
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from src.learning.replay_buffer import ReplayBuffer

CHUNK_PREFIX = "chunk"


def samples_to_arrays(samples: List[Dict]) -> Dict[str, np.ndarray]:
    """Stack a list of experience dicts into contiguous numpy arrays."""
    if not samples:
        raise ValueError("Cannot serialize an empty sample list.")
    return {
        "input": np.stack([s["input"] for s in samples]).astype(np.float32),
        "move_index": np.array([s["move_index"] for s in samples], dtype=np.int16),
        "legal_packed": np.stack([s["legal_packed"] for s in samples]).astype(np.uint64),
        "value": np.array([s["value"] for s in samples], dtype=np.float32),
        "version": np.array([s["version"] for s in samples], dtype=np.int32),
    }


def write_chunk(path: str, samples: List[Dict]) -> None:
    """Serialize a list of samples to a single .npz chunk file."""
    np.savez_compressed(path, **samples_to_arrays(samples))


def read_chunk(path: str) -> List[Dict]:
    """Deserialize a .npz chunk into a list of experience dicts."""
    samples = []
    with np.load(path) as z:
        for i in range(len(z["input"])):
            samples.append(
                {
                    "input": z["input"][i],
                    "move_index": z["move_index"][i],
                    "legal_packed": z["legal_packed"][i],
                    "value": z["value"][i],
                    "version": z["version"][i],
                }
            )
    return samples


def chunk_path(dir_path: str, index: int) -> str:
    return os.path.join(dir_path, f"{CHUNK_PREFIX}_{index:06d}.npz")


def list_chunks(dir_path: str) -> List[str]:
    if not os.path.isdir(dir_path):
        return []
    files = sorted(f for f in os.listdir(dir_path) if f.startswith(CHUNK_PREFIX) and f.endswith(".npz"))
    return [os.path.join(dir_path, f) for f in files]


class ChunkWriter:
    """Accumulates experiences and flushes them to chunk files."""

    def __init__(self, dir_path: str, chunk_size: int = 10_000) -> None:
        self.dir_path = dir_path
        self.chunk_size = chunk_size
        self.pending: List[Dict] = []
        self.chunk_index = 0
        os.makedirs(self.dir_path, exist_ok=True)

    def add(self, samples: List[Dict]) -> None:
        self.pending.extend(samples)
        while len(self.pending) >= self.chunk_size:
            batch = self.pending[: self.chunk_size]
            self.pending = self.pending[self.chunk_size:]
            self._flush(batch)

    def _flush(self, batch: List[Dict]) -> None:
        path = chunk_path(self.dir_path, self.chunk_index)
        write_chunk(path, batch)
        self.chunk_index += 1

    def flush(self) -> None:
        """Flush any remaining samples, creating the final partial chunk."""
        if self.pending:
            self._flush(self.pending)
            self.pending = []


class SelfPlayDataset(Dataset):
    """torch Dataset over a set of chunk files.

    ``__getitem__`` returns tensors shaped for the network:

        input        (18, 8, 8) float32
        move_index   scalar int64   (policy target index)
        legal_packed (64,) uint64
        value        scalar float32
        version      scalar int64
    """

    def __init__(self, chunk_files: List[str]) -> None:
        self.chunk_files = list(chunk_files)
        self._samples: Optional[List[Dict]] = None

    def __len__(self) -> int:
        return sum(len(read_chunk(f)) for f in self.chunk_files)

    def _load_all(self) -> List[Dict]:
        if self._samples is None:
            samples: List[Dict] = []
            for f in self.chunk_files:
                samples.extend(read_chunk(f))
            self._samples = samples
        return self._samples

    def __getitem__(self, idx: int):
        s = self._load_all()[idx]
        return {
            "input": torch.from_numpy(np.ascontiguousarray(s["input"])),
            "move_index": torch.tensor(int(s["move_index"]), dtype=torch.long),
            "legal_packed": torch.from_numpy(s["legal_packed"].astype(np.int64)),
            "value": torch.tensor(float(s["value"]), dtype=torch.float32),
            "version": torch.tensor(int(s["version"]), dtype=torch.long),
        }

    @staticmethod
    def from_replay_buffer(buffer: ReplayBuffer) -> "SelfPlayDataset":
        data = buffer.to_numpy()
        samples = []
        for i in range(len(data["input"])):
            samples.append(
                {
                    "input": data["input"][i],
                    "move_index": data["move_index"][i],
                    "legal_packed": data["legal_packed"][i],
                    "value": data["value"][i],
                    "version": data["version"][i],
                }
            )
        return _SamplesDataset(samples)


class _SamplesDataset(Dataset):
    """In-memory dataset backed by a list of samples (test helper)."""

    def __init__(self, samples: List[Dict]) -> None:
        self._samples = samples

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int):
        s = self._samples[idx]
        return {
            "input": torch.from_numpy(np.ascontiguousarray(s["input"])),
            "move_index": torch.tensor(int(s["move_index"]), dtype=torch.long),
            "legal_packed": torch.from_numpy(s["legal_packed"].astype(np.int64)),
            "value": torch.tensor(float(s["value"]), dtype=torch.float32),
            "version": torch.tensor(int(s["version"]), dtype=torch.long),
        }