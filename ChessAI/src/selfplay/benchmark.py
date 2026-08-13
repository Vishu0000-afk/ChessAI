"""Benchmark mode: measures throughput scientifically instead of guessing.

Runs headless self-play (training disabled) for a fixed number of games and
reports games/sec, moves/sec, positions (inference)/sec, plus a separate
training-throughput measurement (samples/sec) on the collected replay data.
"""

from __future__ import annotations

import logging
import time
from dataclasses import replace

import torch

from src.learning.replay_buffer import ReplayBuffer
from src.selfplay.coordinator import SelfPlayCoordinator

logger = logging.getLogger(__name__)


def run_benchmark(config, games: int) -> dict:
    """Execute the benchmark and return a results dict."""
    bench_config = replace(
        config,
        game_mode="benchmark",
        games_total=int(games),
        train_enabled=False,
        evaluate_enabled=False,
        auto_resume=False,
        checkpoint_every_n_games=0,
    )

    coordinator = SelfPlayCoordinator(bench_config)
    stats = coordinator.run()

    s = stats.snapshot()
    results = {
        "games": s["games"],
        "games_per_sec": s["games_per_sec"],
        "moves_per_sec": s["moves_per_sec"],
        "positions_per_sec": s["samples_per_sec"],  # inference positions/sec
        "samples": s["samples"],
        "avg_moves_per_game": s["avg_moves"],
        "device": bench_config.device,
        "num_workers": bench_config.num_workers,
        "concurrency": bench_config.self_play_concurrency,
        "inference_max_batch": bench_config.inference_max_batch,
    }

    # Peak VRAM report for CUDA runs (device is cuda even on CPU-only builds
    # when explicitly requested, so guard on torch.cuda.is_available()).
    if bench_config.device.startswith("cuda") and torch.cuda.is_available():
        results["vram_allocated_gb"] = torch.cuda.max_memory_allocated() / 1e9
        results["vram_reserved_gb"] = torch.cuda.max_memory_reserved() / 1e9

    # Separate training-throughput micro-benchmark on the collected data.
    if len(coordinator.replay_buffer) >= bench_config.batch_size:
        results["training"] = _measure_training_throughput(coordinator, bench_config)

    _print_report(results)
    return results


def _measure_training_throughput(coordinator, config) -> dict:
    buffer = coordinator.replay_buffer
    trainer = coordinator.learner.trainer
    steps = max(10, min(config.training_steps, len(buffer) // config.batch_size))
    start = time.perf_counter()
    trainer.train(buffer, steps)
    elapsed = max(time.perf_counter() - start, 1e-9)
    samples = steps * config.batch_size
    return {
        "steps": steps,
        "samples_per_sec": samples / elapsed,
        "seconds": elapsed,
    }


def _print_report(results: dict) -> None:
    print("\n================= ChessAI Benchmark =================")
    print(f"Games:              {results['games']:>12,}")
    print(f"Games/sec:          {results['games_per_sec']:>12.1f}")
    print(f"Moves/sec:          {results['moves_per_sec']:>12.1f}")
    print(f"Positions/sec:      {results['positions_per_sec']:>12.1f}")
    print(f"Samples:            {results['samples']:>12,}")
    print(f"Avg game:           {results['avg_moves_per_game']:>12.1f} moves")
    print(f"Device:             {results['device']:>12s}")
    print(f"Workers:            {results['num_workers']:>12d}")
    print(f"Games in flight:    {results['concurrency']:>12d}")
    print(f"Inference max batch:{results['inference_max_batch']:>12d}")
    if "vram_allocated_gb" in results:
        print(f"VRAM peak (alloc):  {results['vram_allocated_gb']:>12.2f} GB")
        print(f"VRAM peak (resv):   {results['vram_reserved_gb']:>12.2f} GB")
    if "training" in results:
        t = results["training"]
        print(f"Train samples/sec:  {t['samples_per_sec']:>12.1f}")
        print(f"Train steps:        {t['steps']:>12d}")
    print("=====================================================")