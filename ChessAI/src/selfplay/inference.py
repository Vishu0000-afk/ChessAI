"""Shared batched neural inference for parallel self-play.

One GPU model lives in the coordinator process. CPU self-play worker
processes never own a model; they send batches of encoded positions over a
request queue and receive ``(logits, values)`` on a result queue. The server
thread accumulates requests and runs batched forwards (mixed precision,
eval mode, no gradients) so the GPU stays busy with large batches.

Training and inference share a lock so they never race on the same module.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from src.learning.encoding import encode_board_batch

logger = logging.getLogger(__name__)

ModelId = str
DEFAULT_MODEL_ID = "current"


class InferenceServer(threading.Thread):
    """Thread that services batched inference requests from worker processes."""

    def __init__(
        self,
        models: Dict[str, torch.nn.Module],
        request_queue,
        result_queue,
        device: str = "cpu",
        max_batch: int = 512,
        use_mixed_precision: bool = True,
        poll_seconds: float = 0.001,
        compute_lock: Optional[threading.Lock] = None,
    ) -> None:
        super().__init__(daemon=True)
        self.models = models
        self.request_queue = request_queue
        self.result_queue = result_queue
        self.device = torch.device(device)
        self.max_batch = max_batch
        self.use_mixed_precision = use_mixed_precision
        self.poll_seconds = poll_seconds
        self.compute_lock = compute_lock
        self._stop = threading.Event()
        self._inflight = 0
        self.worker_queues: Dict[int, object] = {}

        for model in models.values():
            model.eval()
            model.to(self.device)

    def register_model(self, model_id: ModelId, model: torch.nn.Module) -> None:
        model.eval()
        model.to(self.device)
        self.models[model_id] = model

    def register_worker(self, worker_id: int, result_queue) -> None:
        """Associate a worker id with the queue its results must be routed to."""
        self.worker_queues[worker_id] = result_queue

    def stop(self) -> None:
        self._stop.set()

    # ------------------------------------------------------------------
    def run(self) -> None:
        while not self._stop.is_set():
            batch = self._drain()
            if batch:
                self._forward(batch)
            else:
                time.sleep(self.poll_seconds)
        # Drain anything still pending on shutdown.
        final = self._drain()
        if final:
            self._forward(final)

    def _drain(self) -> List[Tuple[int, ModelId, np.ndarray, int]]:
        batch: List[Tuple[int, ModelId, np.ndarray, int]] = []
        while True:
            try:
                item = self.request_queue.get_nowait()
            except Exception:
                break
            batch.append(item)
        return batch

    # ------------------------------------------------------------------
    def _forward(self, batch: List[Tuple[int, ModelId, np.ndarray, int]]) -> None:
        by_model: Dict[ModelId, List[Tuple[int, np.ndarray, int]]] = {}
        for rid, model_id, arr, worker_id in batch:
            by_model.setdefault(model_id, []).append((rid, arr, worker_id))
        for model_id, reqs in by_model.items():
            self._forward_model(model_id, reqs)

    def _forward_model(
        self, model_id: ModelId, reqs: List[Tuple[int, np.ndarray, int]]
    ) -> None:
        model = self.models[model_id]
        acc: List[np.ndarray] = []
        meta: List[Tuple[int, int, int, int]] = []  # (rid, offset, count, worker_id)
        acc_total = 0
        for rid, arr, worker_id in reqs:
            n = len(arr)
            if acc_total + n > self.max_batch and acc:
                self._emit(model, acc, meta)
                acc, meta, acc_total = [], [], 0
            acc.append(arr)
            meta.append((rid, acc_total, n, worker_id))
            acc_total += n
        if acc:
            self._emit(model, acc, meta)

    def _emit(
        self, model: torch.nn.Module, acc: List[np.ndarray], meta: List[Tuple[int, int, int, int]]
    ) -> None:
        self._inflight += 1
        x = torch.from_numpy(np.concatenate(acc, axis=0)).to(self.device)
        lock = self.compute_lock or _nullctx()
        with lock, torch.no_grad():
            with torch.autocast(
                device_type=self.device.type,
                enabled=self.use_mixed_precision and self.device.type in ("cuda", "cpu"),
                dtype=torch.float16 if self.device.type == "cuda" else torch.bfloat16,
            ):
                logits, value = model(x)
        logits_np = logits.detach().float().cpu().numpy()
        value_np = value.detach().float().cpu().numpy().reshape(-1)
        for rid, off, n, worker_id in meta:
            rq = self.worker_queues.get(worker_id) or self.result_queue
            rq.put((rid, (logits_np[off:off + n].copy(), value_np[off:off + n].copy())))
        self._inflight -= 1


class InferenceClient:
    """Queue-based client used by self-play workers.

    Conforms to the same ``predict_batch`` protocol as ``LocalPredictor`` so a
    ``NeuralAgent`` can run identically in-process or against the shared server.
    """

    def __init__(self, request_queue, result_queue, worker_id: Optional[int] = None) -> None:
        self.request_queue = request_queue
        self.result_queue = result_queue
        self.worker_id = worker_id
        self._counter = 0
        self._cache: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

    def predict_batch(self, model_id: ModelId, boards) -> Tuple[np.ndarray, np.ndarray]:
        encoded = encode_board_batch(boards)
        rid = self._next_id()
        # The server routes the reply to this client's queue (via worker_id),
        # so parallel workers never consume each other's results.
        self.request_queue.put((rid, model_id, encoded, self.worker_id))
        return self._await(rid)

    def _next_id(self) -> int:
        self._counter += 1
        return self._counter

    def _await(self, rid: int) -> Tuple[np.ndarray, np.ndarray]:
        while True:
            if rid in self._cache:
                return self._cache.pop(rid)
            got_id, result = self.result_queue.get()
            if got_id == rid:
                return result
            self._cache[got_id] = result


class _nullctx:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False