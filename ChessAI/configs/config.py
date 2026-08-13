"""Simple engine/GUI configuration.

Deliberately plain module-level constants plus a small dataclass —
no config framework, no file parsing. Import what you need.
"""

from __future__ import annotations

from dataclasses import dataclass

import chess

# --- Engine defaults ---
AI_COLOR: bool = chess.BLACK
SEARCH_DEPTH: int = 3
USE_TRANSPOSITION_TABLE: bool = True
USE_MOVE_ORDERING: bool = True

# --- GUI defaults ---
WINDOW_SIZE: int = 640
SQUARE_SIZE: int = WINDOW_SIZE // 8
FPS: int = 30


@dataclass
class EngineConfig:
    """Bundled engine configuration, useful for passing around as one object."""

    ai_color: bool = AI_COLOR
    search_depth: int = SEARCH_DEPTH
    use_transposition_table: bool = USE_TRANSPOSITION_TABLE
    use_move_ordering: bool = USE_MOVE_ORDERING
