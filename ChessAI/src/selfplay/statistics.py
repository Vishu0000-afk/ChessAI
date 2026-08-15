"""Self-play statistics and the live dashboard.

Statistics are accumulated in the coordinator and rendered as a compact
periodic block (not thousands of log lines).
"""

from __future__ import annotations

import logging
import time
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class SelfPlayStats:
    """Accumulated counters for a self-play run."""

    def __init__(self) -> None:
        self.games = 0
        self.moves = 0
        self.samples = 0
        self.white_wins = 0
        self.black_wins = 0
        self.draws = 0
        self.training_steps = 0
        self.model_version = 0
        self.promotions = 0
        self.replay_size = 0
        self.start_time = time.monotonic()
        self.last_time = self.start_time
        self.train_seconds = 0.0

    def add_train_seconds(self, seconds: float) -> None:
        """Accumulate wall-clock spent on learning cycles (not game play)."""
        self.train_seconds += seconds

    def record_game(self, result: str, moves: int, samples: int) -> None:
        self.games += 1
        self.moves += moves
        self.samples += samples
        if result == "1-0":
            self.white_wins += 1
        elif result == "0-1":
            self.black_wins += 1
        else:
            self.draws += 1

    def rates(self) -> Dict[str, float]:
        elapsed = max(time.monotonic() - self.start_time, 1e-9)
        play_elapsed = max(elapsed - self.train_seconds, 1e-9)
        return {
            "games_per_sec": self.games / elapsed,
            "play_games_per_sec": self.games / play_elapsed,
            "moves_per_sec": self.moves / play_elapsed,
            "samples_per_sec": self.samples / play_elapsed,
        }

    def result_rates(self) -> Dict[str, float]:
        total = max(self.games, 1)
        return {
            "white": self.white_wins / total,
            "black": self.black_wins / total,
            "draws": self.draws / total,
        }

    def snapshot(self) -> Dict[str, object]:
        rates = self.rates()
        rr = self.result_rates()
        return {
            "games": self.games,
            "games_per_sec": rates["games_per_sec"],
            "play_games_per_sec": rates["play_games_per_sec"],
            "moves_per_sec": rates["moves_per_sec"],
            "samples_per_sec": rates["samples_per_sec"],
            "moves": self.moves,
            "samples": self.samples,
            "white_wins": self.white_wins,
            "black_wins": self.black_wins,
            "draws": self.draws,
            "white_rate": rr["white"],
            "black_rate": rr["black"],
            "draw_rate": rr["draws"],
            "training_steps": self.training_steps,
            "model_version": self.model_version,
            "promotions": self.promotions,
            "replay_size": self.replay_size,
            "avg_moves": self.moves / max(self.games, 1),
            "elapsed_seconds": time.monotonic() - self.start_time,
        }


class Dashboard:
    """Periodically prints a clean one-block status panel."""

    def __init__(self, interval_seconds: float = 2.0, stream=None) -> None:
        self.interval = interval_seconds
        self.stream = stream or __import__("sys").stdout
        self._last = 0.0

    def maybe_print(self, stats: SelfPlayStats, now: Optional[float] = None) -> None:
        now = now or time.monotonic()
        if now - self._last < self.interval:
            return
        self._last = now
        self._print(stats)

    def final(self, stats: SelfPlayStats) -> None:
        self._print(stats)

    def _print(self, stats: SelfPlayStats) -> None:
        s = stats.snapshot()
        rates = s["games_per_sec"]
        panel = (
            "\n============== ChessAI Self-Play ==============\n"
            f"Games:        {s['games']:>12,}\n"
            f"Games/sec:    {rates:>12.1f}\n"
            f"Play g/sec:   {s['play_games_per_sec']:>12.1f}\n"
            f"Moves/sec:    {s['moves_per_sec']:>12.1f}\n"
            f"Samples:      {s['samples']:>12,}\n"
            f"Training:     {s['training_steps']:>12,}\n"
            f"White wins:   {s['white_rate'] * 100:>11.1f}%\n"
            f"Black wins:   {s['black_rate'] * 100:>11.1f}%\n"
            f"Draws:        {s['draw_rate'] * 100:>11.1f}%\n"
            f"Avg game:     {s['avg_moves']:>12.1f} moves\n"
            f"Model:        v{s['model_version']:>11,}\n"
            f"Replay buf:   {s['replay_size']:>12,}\n"
            f"Elapsed:      {s['elapsed_seconds']:>12.1f}s\n"
            "================================================"
        )
        self.stream.write(panel + "\n")
        self.stream.flush()