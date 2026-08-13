# ChessAI

A clean, modular Python chess engine — V1 foundation. Plays classical
alpha-beta search against a human through a PyGame GUI. Built to be
extended later into a neural-network / MCTS engine without a rewrite.

## Project Purpose

This is the **V1 foundation** of a chess AI project. It deliberately
contains **no machine learning** — just a correct, well-tested classical
engine (material + piece-square evaluation, minimax with alpha-beta
pruning, move ordering, transposition table) plus a playable GUI.
Later phases will swap the classical evaluator for a neural network and
add self-play / MCTS, without needing to touch the board, GUI, or
overall architecture.

## Architecture

```text
src/
├── main.py                 # Entry point — launches the GUI
├── engine/
│   ├── board.py             # Wraps python-chess; clean interface for the rest of the engine
│   ├── evaluator.py         # Material + piece-square + mobility + king-safety scoring
│   ├── move_ordering.py     # Captures/promotions/checks-first heuristics for pruning
│   ├── search.py            # Negamax with alpha-beta pruning
│   ├── transposition.py     # Zobrist-hash-keyed transposition table
│   └── engine.py            # ChessEngine — coordinates the above, exposes get_best_move()
├── gui/
│   ├── board_renderer.py    # Draws the board/pieces/highlights (PyGame)
│   ├── input_handler.py     # Mouse clicks -> chess.Move
│   └── chess_gui.py         # Main app loop: Human vs AI
└── utils/
    └── logging_config.py    # Standard logging setup

configs/config.py            # AI color, search depth, feature toggles
tests/                       # pytest suite for board, evaluator, search, engine
models/ data/ checkpoints/   # Reserved for future ML phases (currently empty)
```

**Design principle:** `python-chess` handles all chess rules (legality,
check/checkmate/draw detection, Zobrist hashing). Nothing here
reimplements chess rules from scratch. Each engine module has a single
responsibility and a narrow interface, so any one piece (e.g. the
evaluator) can be swapped later without touching the others.

## Installation

Requires Python 3.12.

```bash
pip install -r requirements.txt
```

## How to Run

```bash
python src/main.py
```

This launches the PyGame window. You play White; the engine plays Black
by default. Click a piece, then click a destination square. Legal
moves for the selected piece are highlighted; captures show as rings.
Press `R` at any time to restart the game.

## How to Run Tests

```bash
pytest
```

or with verbose output:

```bash
pytest -v
```

31 tests cover the board wrapper, evaluator, search, and high-level
engine, including checkmate-in-one, stalemate, illegal-move rejection,
and transposition-table on/off behavior. Tests are deterministic (no
randomness in search or evaluation).

## Configuration

Edit `configs/config.py`:

```python
AI_COLOR = chess.BLACK          # Which side the engine plays
SEARCH_DEPTH = 3                # Plies searched per move
USE_TRANSPOSITION_TABLE = True
USE_MOVE_ORDERING = True
```

Search depth 3 is snappy; depth 4-5 is noticeably stronger but slower.

## Current Capabilities

- Legal chess play via `python-chess` (all rules: castling, en passant,
  promotion, threefold repetition, fifty-move rule, insufficient material).
- Classical evaluation: material, piece-square tables, mobility, basic
  king-safety (pawn shield).
- Negamax search with alpha-beta pruning, configurable depth.
- Move ordering (captures/promotions/checks first) for faster pruning.
- Transposition table (Zobrist-hash keyed) — can be toggled off.
- Engine statistics: nodes searched, search time, NPS, TT hits.
- PyGame GUI: click-to-move, legal-move highlighting, check/checkmate/
  draw display, restart.

## Future Development Roadmap

This foundation is designed so the following can be added without
restructuring existing code:

1. **Neural network evaluator** — replace `Evaluator.evaluate()` with a
   learned value function; the `Search` module only depends on the
   evaluator's public `evaluate(board) -> int` interface.
2. **Self-play data generation** — use `ChessEngine` + `Board` to
   generate games, storing them under `data/`.
3. **Policy + value network training** — `models/` and `checkpoints/`
   are reserved for this phase.
4. **MCTS** — a new search module implementing Monte Carlo Tree Search,
   selectable alongside the existing alpha-beta search.
5. **Stronger classical heuristics** (optional, in parallel) — better
   king safety, pawn structure, iterative deepening, quiescence search.
