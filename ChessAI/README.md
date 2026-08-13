# ChessAI

A modular Python chess AI that combines a **classical alpha-beta engine**
(material + piece-square evaluation, transposition table, move ordering)
with an **AlphaZero-style neural network** and **parallel GPU self-play**
with continuous learning.

```
python src/main.py --mode self_play --games 1000000 --workers 8 --device cuda
```

Plays classical search against a human through a PyGame GUI, or trains a
policy/value network headlessly through AI-vs-AI self-play.

## Modes

| Mode           | Command                                                        | Description                                             |
| -------------- | -------------------------------------------------------------- | ------------------------------------------------------- |
| `human_vs_ai`  | `python src/main.py`                                           | PyGame GUI, human vs classical / neural / random bot    |
| `self_play`    | `python src/main.py --mode self_play --games 1000000`          | Headless AI-vs-AI training with continuous learning     |
| `benchmark`    | `python src/main.py --mode benchmark --games 10000`            | Headless throughput benchmark (no training)             |

All modes auto-detect the GPU (`--device auto` = CUDA when available).

## Architecture

```text
src/
├── main.py                  # Entry point — CLI for all three modes
├── engine/                  # Classical alpha-beta engine
│   ├── board.py             # Board wrapper around python-chess
│   ├── evaluator.py         # Material + PST + mobility + king safety
│   ├── move_ordering.py     # MVV-LVA captures / promos / checks, TT-move first
│   ├── search.py            # Negamax with alpha-beta pruning
│   ├── transposition.py     # Zobrist-keyed transposition table
│   └── engine.py            # ChessEngine facade + statistics
├── learning/                # Neural network stack
│   ├── encoding.py          # 18x8x8 position planes; 4096-move policy space
│   ├── network.py           # Conv trunk + residual blocks + policy/value heads
│   ├── replay_buffer.py     # Pre-allocated numpy ring buffer
│   ├── dataset.py           # Chunked .npz storage + torch Dataset
│   ├── trainer.py           # Masked-CE policy + MSE value, AMP mixed precision
│   ├── learner.py           # Training / checkpoint / resume orchestration
│   └── model_manager.py     # Versioned checkpoints + latest.pth resume
├── selfplay/                # Multi-process producer/consumer training loop
│   ├── coordinator.py       # Workers + replay + train + evaluate + promote
│   ├── worker.py            # BatchedGameSimulator — lockstep parallel games
│   ├── inference.py         # One shared GPU model, queue-based batching
│   ├── evaluator.py         # Candidate vs previous-model match (ELO estimate)
│   ├── benchmark.py         # Throughput harness
│   └── statistics.py        # SelfPlayStats + live dashboard
├── agents/                  # ChessAgent interface + classical/neural/random bots
│   ├── base.py              # Abstract ChessAgent (select_move / predict_batch)
│   ├── classical_engine.py  # Wraps ChessEngine as a bot
│   ├── neural_agent.py      # Move selection from masked policy logits
│   ├── predictor.py         # LocalPredictor (in-process batched inference)
│   └── random_agent.py      # Uniform-random baseline
├── gui/                     # PyGame renderer / input handler / app loop
└── utils/
    └── logging_config.py    # Standard logging setup

configs/config.py            # All tunables (engine, NN, self-play, checkpointing)
tests/                       # pytest suite (61 tests)
models/ data/ checkpoints/   # Outputs (gitignored)
```

**Design principle:** `python-chess` handles all chess rules (legality,
check/checkmate/draw detection, Zobrist hashing) — nothing reimplements rules
from scratch. The `ChessAgent` interface abstracts the decision-maker, so the
classical engine, the neural agent, and a queue-based inference client are all
drop-in interchangeable.

## Installation

Requires Python 3.12.

```bash
pip install -r requirements.txt
```

For GPU training, install a CUDA-enabled PyTorch (see
[pytorch.org](https://pytorch.org/get-started/locally/)); `torch.cuda.is_available()`
is auto-detected.

## Usage

### Human vs AI (GUI)

```bash
python src/main.py
```

You play White; the engine plays Black by default. Click a piece, then click a
destination square. Legal moves for the selected piece are highlighted (captures
show as rings). Press `R` to restart.

Choose the opponent with `--ai-agent classical|neural|random` (classical by
default) and its depth with `--depth 4`.

### Self-play (continuous learning)

```bash
python src/main.py --mode self_play --games 1000000 --workers 8 --device cuda
```

- CPU **workers** play games in lockstep (`--concurrency` games in flight each).
- Workers never own a model: every position batch is sent to a shared
  `InferenceServer` thread, which runs one batched fp16 forward on the GPU.
- Experiences stream into a replay buffer; every `--train-every` games the
  learner trains the network (masked cross-entropy on the played move, MSE on
  the game result).
- The trained candidate is then matched against the previous model
  (`--evaluate-games` games, deterministic argmax); it is **promoted** only if
  it scores at least `promotion_min_score` (default 0.55), otherwise weights are
  reverted.
- Temperature starts at `--temperature 1.0` and decays toward `temp_final` over
  `temp_decay_games` games.
- Progress prints every few seconds; checkpoints, replay buffer, and chunked
  training data persist under `checkpoints/` and `data/self_play/`, and resume
  automatically (`--no-resume` to start fresh).

### Benchmark

```bash
python src/main.py --mode benchmark --games 10000
```

Measures games/sec, moves/sec, and inference positions/sec, plus a separate
training-throughput measurement on the collected data.

## Configuration

All tunables live in `configs/config.py`. Highlights:

```python
GAME_MODE = "human_vs_ai"     # "human_vs_ai" | "self_play" | "benchmark"
SEARCH_DEPTH = 3              # classical engine plies per move

NN_CONV_CHANNELS = 64         # network width
NN_RES_BLOCKS = 2             # residual blocks
REPLAY_BUFFER_SIZE = 1_000_000
TRAIN_EVERY_N_GAMES = 100
BATCH_SIZE = 512
TRAINING_STEPS = 100
LEARNING_RATE = 1e-3
USE_MIXED_PRECISION = True    # fp16 on CUDA, bf16 on CPU

EVALUATE_GAMES = 40           # candidate match games before promotion
PROMOTION_MIN_SCORE = 0.55
CHECKPOINT_EVERY_N_GAMES = 1_000
```

Search depth 3 is snappy; depth 4–5 is noticeably stronger but slower.

## Testing

```bash
pytest
# or verbose:
pytest -v
```

61 tests cover the board wrapper, classical evaluator/search/engine, agents,
encoding, replay buffer, dataset, trainer, checkpoint/resume, and an end-to-end
self-play smoke run (including resume-after-interruption).

## Data & Checkpoints

| Path                  | Contents                                                    |
| --------------------- | ----------------------------------------------------------- |
| `checkpoints/`        | `model_000001.pth`, ... versioned weights + optimizer state |
| `checkpoints/latest.pth` | Always mirrors the newest checkpoint (for resume)       |
| `data/self_play/`     | `chunk_XXXXXX.npz` serialized experiences, `replay_buffer.npz` |
| `models/`             | Reserved                                                    |

All are gitignored; checkpoints and data can be deleted freely between runs.

## Roadmap

- **MCTS search** — a Monte Carlo Tree Search module consuming the same
  policy/value network, selectable alongside alpha-beta.
- **Stronger classical heuristics** — iterative deepening, quiescence search,
  killer moves / history heuristic.
- **Multi-GPU / larger networks** — the pipeline already batches across workers;
  the inference server scales with batch size.
