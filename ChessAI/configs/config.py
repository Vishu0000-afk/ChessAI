"""Engine / GUI / self-play / learning configuration.

Deliberately plain module-level constants plus a small dataclass —
no config framework, no file parsing. Import what you need.

GAME_MODE selects the top-level operation:
    "human_vs_ai"  -> PyGame GUI, Human vs AI (unchanged behaviour)
    "self_play"    -> headless AI-vs-AI self-play with continuous learning
    "benchmark"    -> headless throughput benchmark (no training)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, Optional

import chess

# =============================================================================
# Top-level mode
# =============================================================================
GAME_MODE: str = "human_vs_ai"  # "human_vs_ai" | "self_play" | "benchmark"

# --- Engine defaults (classical engine, also used as a baseline agent) ---
AI_COLOR: bool = chess.BLACK
SEARCH_DEPTH: int = 3
USE_TRANSPOSITION_TABLE: bool = True
USE_MOVE_ORDERING: bool = True

# Agent used by the AI in HUMAN_VS_AI mode: "classical" | "neural" | "random"
AI_AGENT: str = "classical"

# --- GUI defaults ---
WINDOW_SIZE: int = 640
SQUARE_SIZE: int = WINDOW_SIZE // 8
FPS: int = 30
AI_MOVE_DELAY: float = 2.0  # seconds waited after the human move before the AI plays
ANIM_DURATION: float = 0.5  # seconds a piece takes to slide between squares

# =============================================================================
# Self-play
# =============================================================================
# Number of worker processes. Defaults to the available CPU count (capped).
NUM_WORKERS: int = min(16, os.cpu_count() or 1)
# Games kept in flight concurrently inside a single worker process. Combined
# with NUM_WORKERS this is the effective inference batch size.
SELF_PLAY_CONCURRENCY: int = 4
# Games each worker is asked to produce; the coordinator stops once the
# global target (games_total) is reached.
GAMES_TOTAL: int = 10_000

# Exploration temperature. Values >= 1 encourage exploration; decays linearly
# towards TEMP_FINAL over TEMP_DECAY_GAMES self-play games. Decay should span
# roughly 60-80% of the total run so exploration does not die just as the
# model starts to strengthen (a common cause of self-play color collapse).
TEMPERATURE: float = 1.0
TEMP_FINAL: float = 0.3
TEMP_DECAY_GAMES: int = 20_000

# Hard cap on moves per game (avoids endless shuffling). Games that hit the
# cap count as a draw for learning targets and statistics.
MAX_GAME_MOVES: int = 400

# Maximum share of self-play games that may count as draws. When the running
# draw rate reaches this ceiling, excess draw games are converted into wins
# for the color that is currently winning less (see coordinator). This keeps
# the training signal decisive and rebalances the white/black distribution.
DRAW_MAX_RATE: float = 0.2

# Number of uniformly random plies played at the start of every self-play game.
# Stops two identical near-deterministic policies from following one fixed line
# straight into a repetition draw (the "self-play draw collapse").
SELF_PLAY_RANDOM_OPEN_PLIES: int = 8

# Dirichlet noise mixed into the sampled policy during self-play (AlphaZero-style:
# eps * Dir(alpha) + (1 - eps) * policy). Keeps play stochastic after the opening
# so deterministic models cannot lock into a repeating position cycle.
DIRICHLET_EPSILON: float = 0.25
DIRICHLET_ALPHA: float = 0.03

# =============================================================================
# Neural network / learning
# =============================================================================
# Network architecture.
NN_INPUT_CHANNELS: int = 18
NN_CONV_CHANNELS: int = 64
NN_RES_BLOCKS: int = 2  # residual blocks stacked on the convolutional trunk

# Batched inference tuning.
INFERENCE_MAX_BATCH: int = 512  # max positions per GPU/CPU forward pass
INFERENCE_POLL_SECONDS: float = 0.001

# Training.
REPLAY_BUFFER_SIZE: int = 1_000_000
TRAIN_EVERY_N_GAMES: int = 100
BATCH_SIZE: int = 512
TRAINING_STEPS: int = 100
LEARNING_RATE: float = 1e-3
WEIGHT_DECAY: float = 1e-4
# How many games a worker plays before returning its experiences to the
# coordinator (keeps IPC churn low).
EXPERIENCE_FLUSH_GAMES: int = 1

# Mixed precision: float16 on CUDA, bfloat16 on CPU.
USE_MIXED_PRECISION: bool = True

# Mirror (horizontal file flip) ~50% of each training batch so the network
# learns board symmetry and needs half the data to master it.
USE_MIRROR_AUGMENTATION: bool = True

# =============================================================================
# Checkpoints / model versioning
# =============================================================================
CHECKPOINT_DIR: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "checkpoints")
CHECKPOINT_EVERY_N_GAMES: int = 1_000
# Keep checkpoints keyed by version + a rotating "latest.pth" symlink-style copy.
# On startup the newest checkpoint is loaded (resume after interruption).
AUTO_RESUME: bool = True
MODEL_SAVE_NAME: str = "model_{version:06d}.pth"
LATEST_SAVE_NAME: str = "latest.pth"

# =============================================================================
# Self-play dataset (chunked serialized storage under data/self_play)
# =============================================================================
DATA_DIR: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
SELF_PLAY_DATA_DIR: str = os.path.join(DATA_DIR, "self_play")
DATASET_CHUNK_SIZE: int = 10_000  # samples per chunk file
# Persist the replay buffer to the dataset directory every N games (0 = off).
PERSIST_EVERY_N_GAMES: int = 1_000

# =============================================================================
# Evaluation / promotion
# =============================================================================
EVALUATE_ENABLED: bool = True
EVALUATE_GAMES: int = 40
EVALUATE_CONCURRENCY: int = 8
# Minimum candidate score (wins + 0.5*draws) required to promote over the
# previous model.
PROMOTION_MIN_SCORE: float = 0.55

# =============================================================================
# Statistics dashboard
# =============================================================================
DASHBOARD_INTERVAL_SECONDS: float = 2.0

# =============================================================================
# Device selection
# =============================================================================
def resolve_device(override: Optional[str] = None):
    """Return the torch device string for computation.

    Args:
        override: "auto" | "cpu" | "cuda" | "cuda:0" ...
    """
    if override and override != "auto":
        return override
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


DEVICE: str = resolve_device()


@dataclass
class EngineConfig:
    """Bundled classical-engine configuration."""

    ai_color: bool = AI_COLOR
    search_depth: int = SEARCH_DEPTH
    use_transposition_table: bool = USE_TRANSPOSITION_TABLE
    use_move_ordering: bool = USE_MOVE_ORDERING


@dataclass
class SelfPlayConfig:
    """Bundled self-play + learning configuration.

    The dataclass captures every tunable so it can be snapshotted into
    checkpoints, run in experiments, and overridden on the command line.
    """

    game_mode: str = GAME_MODE
    games_total: int = GAMES_TOTAL
    num_workers: int = NUM_WORKERS
    self_play_concurrency: int = SELF_PLAY_CONCURRENCY
    temperature: float = TEMPERATURE
    temp_final: float = TEMP_FINAL
    temp_decay_games: int = TEMP_DECAY_GAMES
    max_game_moves: int = MAX_GAME_MOVES
    draw_max_rate: float = DRAW_MAX_RATE
    self_play_random_open_plies: int = SELF_PLAY_RANDOM_OPEN_PLIES
    dirichlet_epsilon: float = DIRICHLET_EPSILON
    dirichlet_alpha: float = DIRICHLET_ALPHA

    replay_buffer_size: int = REPLAY_BUFFER_SIZE
    train_every_n_games: int = TRAIN_EVERY_N_GAMES
    batch_size: int = BATCH_SIZE
    training_steps: int = TRAINING_STEPS
    learning_rate: float = LEARNING_RATE
    weight_decay: float = WEIGHT_DECAY
    use_mixed_precision: bool = USE_MIXED_PRECISION
    use_mirror_augmentation: bool = USE_MIRROR_AUGMENTATION

    nn_conv_channels: int = NN_CONV_CHANNELS
    nn_res_blocks: int = NN_RES_BLOCKS

    checkpoint_dir: str = CHECKPOINT_DIR
    checkpoint_every_n_games: int = CHECKPOINT_EVERY_N_GAMES
    auto_resume: bool = AUTO_RESUME

    dataset_dir: str = SELF_PLAY_DATA_DIR
    dataset_chunk_size: int = DATASET_CHUNK_SIZE
    persist_every_n_games: int = PERSIST_EVERY_N_GAMES

    evaluate_enabled: bool = EVALUATE_ENABLED
    evaluate_games: int = EVALUATE_GAMES
    evaluate_concurrency: int = EVALUATE_CONCURRENCY
    promotion_min_score: float = PROMOTION_MIN_SCORE

    inference_max_batch: int = INFERENCE_MAX_BATCH
    device: str = DEVICE
    experience_flush_games: int = EXPERIENCE_FLUSH_GAMES
    dashboard_interval_seconds: float = DASHBOARD_INTERVAL_SECONDS

    train_enabled: bool = True  # set False in benchmark mode
    seed: int = 0

    def as_dict(self) -> Dict[str, object]:
        return field_dict(self)


def field_dict(dc) -> Dict[str, object]:
    return {f.name: getattr(dc, f.name) for f in dc.__dataclass_fields__.values()}
