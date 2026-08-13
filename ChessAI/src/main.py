"""Entry point for ChessAI.

Modes:
    human_vs_ai   launch the PyGame GUI (Human vs AI)   [default]
    self_play     headless AI-vs-AI self-play with continuous learning
    benchmark     headless throughput benchmark (no training)

Examples:
    python src/main.py
    python src/main.py --mode human_vs_ai
    python src/main.py --mode self_play --games 1000000 --workers 8
    python src/main.py --mode benchmark --games 10000
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import replace
from typing import Optional

# Allow running as `python src/main.py` from the project root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import (  # noqa: E402
    AI_AGENT,
    AI_COLOR,
    CHECKPOINT_DIR,
    SEARCH_DEPTH,
    GAME_MODE,
    SelfPlayConfig,
    resolve_device,
)
from src.utils.logging_config import setup_logging  # noqa: E402

logger = logging.getLogger(__name__)


def _parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ChessAI engine")
    parser.add_argument(
        "--mode",
        choices=["human_vs_ai", "self_play", "benchmark"],
        default=GAME_MODE,
        help="Top-level operating mode.",
    )
    parser.add_argument("--games", type=int, default=None, help="Self-play/benchmark game target.")
    parser.add_argument("--workers", type=int, default=None, help="Number of self-play worker processes.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="Compute device.")
    parser.add_argument("--depth", type=int, default=None, help="Classical engine search depth.")
    parser.add_argument("--ai-agent", choices=["classical", "neural", "random"], default=None, help="Human-vs-AI opponent.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for self-play.")
    parser.add_argument("--no-train", action="store_true", help="Self-play without training (data collection only).")
    parser.add_argument("--no-evaluate", action="store_true", help="Disable candidate evaluation/promotion.")
    parser.add_argument("--train-every", type=int, default=None, help="Train every N games.")
    parser.add_argument("--steps", type=int, default=None, help="Training steps per cycle.")
    parser.add_argument("--batch-size", type=int, default=None, help="Training mini-batch size.")
    parser.add_argument("--checkpoint-every", type=int, default=None, help="Checkpoint every N games.")
    parser.add_argument("--temperature", type=float, default=None, help="Self-play exploration temperature.")
    parser.add_argument("--replay-size", type=int, default=None, help="Replay buffer capacity.")
    parser.add_argument("--concurrency", type=int, default=None, help="Games in flight per worker.")
    parser.add_argument("--no-resume", action="store_true", help="Do not resume from the latest checkpoint.")
    return parser.parse_args(argv)


def _selfplay_config(args: argparse.Namespace, train_enabled: bool = True) -> SelfPlayConfig:
    overrides: dict = {
        "game_mode": args.mode,
        "train_enabled": train_enabled and not args.no_train,
        "evaluate_enabled": not args.no_evaluate,
        "device": resolve_device(args.device),
        "auto_resume": not args.no_resume,
    }
    mapping = {
        "games": "games_total",
        "workers": "num_workers",
        "depth": None,
        "ai_agent": None,
        "seed": "seed",
        "train_every": "train_every_n_games",
        "steps": "training_steps",
        "batch_size": "batch_size",
        "checkpoint_every": "checkpoint_every_n_games",
        "temperature": "temperature",
        "replay_size": "replay_buffer_size",
        "concurrency": "self_play_concurrency",
    }
    for arg, field in mapping.items():
        value = getattr(args, arg)
        if value is not None and field is not None:
            overrides[field] = value
    return replace(SelfPlayConfig(), **overrides)


def _launch_human_vs_ai(args: argparse.Namespace) -> None:
    from src.agents.base import ChessAgent
    from src.agents.classical_engine import ClassicalEngineAgent
    from src.agents.random_agent import RandomAgent
    from src.gui.chess_gui import ChessGUI

    agent_type = args.ai_agent or AI_AGENT
    depth = args.depth or SEARCH_DEPTH
    device = resolve_device(args.device)

    if agent_type == "neural":
        agent = _make_neural_agent(device)
        logger.info("Human vs AI using neural agent.")
    elif agent_type == "random":
        agent = RandomAgent()
        logger.info("Human vs AI using random agent.")
    else:
        agent = ClassicalEngineAgent(depth=depth)
        logger.info("Human vs AI using classical engine (depth=%d).", depth)

    gui = ChessGUI(ai_color=AI_COLOR, search_depth=depth, agent=agent)
    gui.run()


def _make_neural_agent(device: str) -> "ChessAgent":
    from src.agents.neural_agent import NeuralAgent
    from src.agents.predictor import LocalPredictor
    from src.learning.model_manager import ModelManager
    from src.learning.network import create_chess_net

    model = create_chess_net()
    manager = ModelManager(CHECKPOINT_DIR)
    meta = manager.resume(model)
    model.eval()
    predictor = LocalPredictor(model, device=device)
    version = meta.version if meta is not None else None
    logger.info("Neural agent loaded (version=%s).", version)
    return NeuralAgent(predictor, temperature=0.0, version=version)


def main(argv: Optional[list] = None) -> int:
    args = _parse_args(argv)
    setup_logging(level=logging.INFO)

    if args.mode == "human_vs_ai":
        _launch_human_vs_ai(args)
        return 0

    if args.mode == "benchmark":
        from src.selfplay.benchmark import run_benchmark

        cfg = _selfplay_config(args, train_enabled=False)
        games = args.games or 10_000
        run_benchmark(cfg, games)
        return 0

    if args.mode == "self_play":
        from src.selfplay.coordinator import SelfPlayCoordinator

        cfg = _selfplay_config(args, train_enabled=True)
        games = args.games or cfg.games_total
        cfg = replace(cfg, games_total=games)
        logger.info("Starting self-play: games=%d workers=%d device=%s", games, cfg.num_workers, cfg.device)
        coordinator = SelfPlayCoordinator(cfg)
        coordinator.run()
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())