"""Self-play coordinator: workers + replay buffer + training + evaluation.

Producer/consumer architecture:

    CPU workers (processes)
          |  experiences
          v
    Replay buffer (main process)
          |
          v
    Trainer (main process, shared compute lock)
          |
          v
    Inference server (single GPU model) <--- workers' batched requests
"""

from __future__ import annotations

import copy
import logging
import multiprocessing as mp
import queue as queue_mod
import sys
import threading
import time
from typing import List, Optional

from src.learning.dataset import ChunkWriter
from src.learning.learner import Learner
from src.learning.model_manager import ModelManager
from src.learning.network import create_chess_net
from src.learning.replay_buffer import ReplayBuffer
from src.selfplay.evaluator import evaluate_models
from src.selfplay.inference import InferenceServer
from src.selfplay.statistics import Dashboard, SelfPlayStats
from src.selfplay.worker import SelfPlayWorker

logger = logging.getLogger(__name__)

REPLAY_BUFFER_FILE = "replay_buffer.npz"


def _ensure_mp_start_method(device: str) -> None:
    """Choose a safe multiprocessing start method.

    Forking a parent that has already initialized CUDA (and a server thread)
    is unsafe, so self-play on CUDA switches to the ``spawn`` start method.
    Everything else keeps the platform default (fork on Linux, spawn on
    Windows).
    """
    try:
        default = mp.get_start_method()
    except AssertionError:
        return
    if default == "fork" and device.startswith("cuda"):
        try:
            mp.set_start_method("spawn", force=True)
            logger.info("Using multiprocessing start method 'spawn' (CUDA).")
        except Exception:
            logger.warning("Could not switch to 'spawn' start method; fork may be unsafe with CUDA.")


def drop_excess_draw(game: dict, draws: int, games: int, draw_max_rate: float) -> bool:
    """Return True when a drawn game should be discarded.

    Draws are allowed while the running draw rate (draws / games) stays below
    ``draw_max_rate``. Once the cap is reached, additional draw games are
    dropped entirely (their samples are not stored and they do not count as
    draws), keeping the training signal decisive without rebalancing colors.
    """
    if game["result"] != "1/2-1/2":
        return False
    draw_rate = draws / games if games else 0.0
    return draw_rate >= draw_max_rate


class SelfPlayCoordinator:
    """Runs headless AI-vs-AI self-play with continuous learning."""

    def __init__(self, config) -> None:
        self.config = config
        self.device = config.device
        self.compute_lock = threading.Lock()
        _ensure_mp_start_method(config.device)

        self.request_queue = mp.Queue()
        self.result_queue = mp.Queue()
        self.experience_queue = mp.Queue()
        self.stop_event = mp.Event()
        self.worker_result_queues = []

        self.model = create_chess_net(
            channels=config.nn_conv_channels, res_blocks=config.nn_res_blocks
        )
        self.replay_buffer = ReplayBuffer(config.replay_buffer_size)
        self.model_manager = ModelManager(config.checkpoint_dir)
        self.learner = Learner(
            model=self.model,
            replay_buffer=self.replay_buffer,
            model_manager=self.model_manager,
            device=self.device,
            batch_size=config.batch_size,
            training_steps=config.training_steps,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            use_mixed_precision=config.use_mixed_precision,
            mirror_augmentation=config.use_mirror_augmentation,
            compute_lock=self.compute_lock,
        )
        self.server: Optional[InferenceServer] = None
        self.workers: List[SelfPlayWorker] = []
        self.stats = SelfPlayStats()
        self.dashboard = Dashboard(self.config.dashboard_interval_seconds)
        self.chunk_writer = ChunkWriter(self.config.dataset_dir, self.config.dataset_chunk_size)

        self.start_games = 0
        self.target_games = self.config.games_total

    # ------------------------------------------------------------------
    def run(self) -> SelfPlayStats:
        try:
            self._resume()
            self._start_server()
            self._start_workers()
            self._loop()
        except KeyboardInterrupt:
            logger.info("Interrupt received; shutting down gracefully...")
        finally:
            self._shutdown()
        return self.stats

    # ------------------------------------------------------------------
    def _resume(self) -> None:
        if self.config.auto_resume:
            meta = self.learner.resume()
            if meta is not None:
                self.start_games = min(meta.games_trained, self.target_games)
                self._load_replay_buffer()
                logger.info(
                    "Resumed: model v%d, games already trained=%d, remaining=%d",
                    meta.version,
                    self.start_games,
                    self.target_games - self.start_games,
                )
        self.stats.model_version = self.learner.version

    def _load_replay_buffer(self) -> None:
        path = f"{self.config.dataset_dir}/{REPLAY_BUFFER_FILE}"
        try:
            self.replay_buffer.load(path)
        except Exception as exc:
            logger.info("No replay buffer to restore (%s).", exc)

    def _save_replay_buffer(self) -> None:
        if len(self.replay_buffer) == 0:
            return
        path = f"{self.config.dataset_dir}/{REPLAY_BUFFER_FILE}"
        self.replay_buffer.save(path)

    # ------------------------------------------------------------------
    def _start_server(self) -> None:
        self.server = InferenceServer(
            models={"current": self.model},
            request_queue=self.request_queue,
            result_queue=self.result_queue,
            device=self.device,
            max_batch=self.config.inference_max_batch,
            use_mixed_precision=self.config.use_mixed_precision,
            compute_lock=self.compute_lock,
        )
        self.server.start()

    def _start_workers(self) -> None:
        remaining = self.target_games - self.start_games
        if remaining <= 0:
            logger.info("All target games already completed; nothing to do.")
            return
        n_workers = min(self.config.num_workers, remaining)
        base, extra = divmod(remaining, n_workers)
        offset = 0
        for w in range(n_workers):
            target = base + (1 if w < extra else 0)
            worker_result_queue = mp.Queue()
            self.worker_result_queues.append(worker_result_queue)
            self.server.register_worker(w, worker_result_queue)
            worker = SelfPlayWorker(
                request_queue=self.request_queue,
                result_queue=worker_result_queue,
                experience_queue=self.experience_queue,
                games_target=target,
                global_start=self.start_games + offset,
                games_global=self.target_games,
                concurrency=self.config.self_play_concurrency,
                temperature=self.config.temperature,
                temp_final=self.config.temp_final,
                temp_decay_games=self.config.temp_decay_games,
                max_game_moves=self.config.max_game_moves,
                model_version=self.learner.version,
                seed=self.config.seed + w,
                stop_event=self.stop_event,
                flush_every=max(1, self.config.experience_flush_games),
                worker_id=w,
            )
            worker.start()
            self.workers.append(worker)
            offset += target
        logger.info("Started %d self-play workers (%d games remaining).", n_workers, remaining)

    # ------------------------------------------------------------------
    def _loop(self) -> None:
        games_since_train = 0
        self.stats.start_time = time.monotonic()
        while True:
            drained = self._drain_experiences()
            games_since_train += drained

            total_done = self.start_games + self.stats.games
            if total_done >= self.target_games:
                break

            if (
                self.config.train_enabled
                and games_since_train >= self.config.train_every_n_games
                and len(self.replay_buffer) >= self.config.batch_size
            ):
                self._learning_cycle()
                games_since_train = 0

            self.stats.replay_size = len(self.replay_buffer)
            self.stats.model_version = self.learner.version
            self.dashboard.maybe_print(self.stats)

            if self.stop_event.is_set():
                break
            time.sleep(0.05)

        # Final cleanup pass of any queued experiences.
        self._drain_experiences()
        self.dashboard.final(self.stats)

    def _drain_experiences(self) -> int:
        drained = 0
        while True:
            try:
                msg = self.experience_queue.get_nowait()
            except queue_mod.Empty:
                break
            for game in msg["games"]:
                if drop_excess_draw(game, self.stats.draws, self.stats.games, self.config.draw_max_rate):
                    self.stats.games += 1
                    self.stats.moves += game["moves"]
                    self.stats.samples += len(game["samples"])
                    drained += 1
                    continue
                self.stats.record_game(game["result"], game["moves"], len(game["samples"]))
                for s in game["samples"]:
                    s.pop("color", None)
                self.replay_buffer.extend(game["samples"])
                self.chunk_writer.add(game["samples"])
                drained += 1
        return drained

    # ------------------------------------------------------------------
    def _learning_cycle(self) -> None:
        # Snapshot the active weights BEFORE training: the previous model.
        prev_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}

        summary = self.learner.train()
        if not summary:
            return
        self.stats.training_steps = self.learner.total_training_steps

        promote = True
        if self.config.evaluate_enabled and self.config.evaluate_games > 0 and self.learner.version > 0:
            promote = self._evaluate_and_decide(prev_state)
        elif self.learner.version == 0:
            promote = True

        if promote:
            self.learner.version += 1
            self.stats.promotions += 1
            logger.info("PROMOTED candidate to model v%d.", self.learner.version)
        else:
            # Reject: restore the previous active weights.
            self.model.load_state_dict(prev_state)
            logger.info("Rejected candidate; keeping model v%d.", self.learner.version)

        self.learner.checkpoint(
            games_trained=self.start_games + self.stats.games,
            version=self.learner.version,
            config=self.config.as_dict(),
            replay_info={"samples": len(self.replay_buffer)},
        )
        self.stats.model_version = self.learner.version
        self.stats.replay_size = len(self.replay_buffer)
        self.stats.training_steps = self.learner.total_training_steps

    def _evaluate_and_decide(self, prev_state) -> bool:
        """Register the previous model, play evaluation games, decide promotion."""
        prev_model = create_chess_net(
            channels=self.config.nn_conv_channels, res_blocks=self.config.nn_res_blocks
        )
        prev_model.load_state_dict(prev_state)
        self.server.register_model("previous", prev_model)

        try:
            result = evaluate_models(
                self.request_queue,
                self.result_queue,
                model_a="current",
                model_b="previous",
                num_games=self.config.evaluate_games,
                concurrency=self.config.evaluate_concurrency,
                max_game_moves=self.config.max_game_moves,
            )
        finally:
            self.server.models.pop("previous", None)

        score = result.score
        promote = score >= self.config.promotion_min_score
        logger.info(
            "Evaluation v%d vs v%d: %d games, candidate score=%.3f, ELO=%.1f -> %s",
            self.learner.version + 1,
            self.learner.version,
            result.games,
            score,
            result.elo_estimate,
            "PROMOTE" if promote else "REJECT",
        )
        return promote

    # ------------------------------------------------------------------
    def _shutdown(self) -> None:
        logger.info("Shutting down self-play coordinator...")
        self.stop_event.set()
        for w in self.workers:
            w.join(timeout=10)
        for w in self.workers:
            if w.is_alive():
                w.terminate()
        self._drain_experiences()

        if self.config.train_enabled and len(self.replay_buffer) >= self.config.batch_size:
            try:
                self._learning_cycle()
            except Exception:
                logger.exception("Final training cycle failed")

        self.chunk_writer.flush()
        self._save_replay_buffer()
        self.dashboard.final(self.stats)

        if self.server is not None:
            self.server.stop()
            self.server.join(timeout=5)
        for q in (self.request_queue, self.result_queue, self.experience_queue):
            q.close()
        for q in self.worker_result_queues:
            q.close()
            q.join_thread()
        logger.info("Self-play finished: %d games.", self.start_games + self.stats.games)