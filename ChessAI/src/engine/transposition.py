"""Dictionary-based transposition table.

Uses python-chess's built-in Zobrist hashing (`board.transposition_key`
or `chess.polyglot.zobrist_hash`) rather than implementing hashing
from scratch. Positions are keyed by Zobrist hash; entries store the
search depth, score, best move, and bound type so the search can
reuse or safely ignore stale/insufficient-depth entries.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional

import chess
import chess.polyglot


class Bound(Enum):
    """Type of score bound stored in a transposition-table entry.

    EXACT: The stored score is the exact minimax value.
    LOWER: The stored score is a lower bound (failed high, beta cutoff).
    UPPER: The stored score is an upper bound (failed low).
    """

    EXACT = "EXACT"
    LOWER = "LOWER"
    UPPER = "UPPER"


@dataclass
class TranspositionEntry:
    """A single transposition-table entry."""

    key: int
    depth: int
    score: int
    best_move: Optional[chess.Move]
    bound: Bound


class TranspositionTable:
    """Simple dictionary-based transposition table keyed by Zobrist hash."""

    def __init__(self, enabled: bool = True) -> None:
        """Create a transposition table.

        Args:
            enabled: If False, all lookups return None and stores are no-ops.
                Allows the search engine to run correctly with the table
                disabled without branching logic elsewhere.
        """
        self.enabled = enabled
        self._table: Dict[int, TranspositionEntry] = {}
        self.hits = 0
        self.stores = 0

    @staticmethod
    def compute_key(board: chess.Board) -> int:
        """Compute the Zobrist hash key for a position."""
        return chess.polyglot.zobrist_hash(board)

    def lookup(self, key: int, depth: int, alpha: int, beta: int) -> Optional[TranspositionEntry]:
        """Look up a usable entry for the given key and required depth.

        Args:
            key: Zobrist hash of the position.
            depth: Minimum search depth required to trust this entry.
            alpha: Current alpha bound (used to decide if entry is usable).
            beta: Current beta bound (used to decide if entry is usable).

        Returns:
            The entry if present and at least as deep as `depth`, else None.
            Note: the caller is responsible for interpreting the bound type
            against alpha/beta; this method returns the raw entry.
        """
        if not self.enabled:
            return None
        entry = self._table.get(key)
        if entry is None:
            return None
        if entry.depth < depth:
            return None
        self.hits += 1
        return entry

    def store(
        self,
        key: int,
        depth: int,
        score: int,
        best_move: Optional[chess.Move],
        bound: Bound,
    ) -> None:
        """Store or overwrite an entry.

        Always-replace strategy: newer, at-least-as-deep searches
        overwrite older shallower entries for simplicity in V1.
        """
        if not self.enabled:
            return
        existing = self._table.get(key)
        if existing is not None and existing.depth > depth:
            # Keep the deeper, more trustworthy existing entry.
            return
        self._table[key] = TranspositionEntry(
            key=key, depth=depth, score=score, best_move=best_move, bound=bound
        )
        self.stores += 1

    def clear(self) -> None:
        """Remove all entries and reset statistics."""
        self._table.clear()
        self.hits = 0
        self.stores = 0

    def __len__(self) -> int:
        return len(self._table)
