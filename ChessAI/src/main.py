"""Entry point: launches the ChessAI PyGame GUI (Human vs AI)."""

from __future__ import annotations

import logging
import os
import sys

# Allow running as `python src/main.py` from the project root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import AI_COLOR, SEARCH_DEPTH  # noqa: E402
from src.gui.chess_gui import ChessGUI  # noqa: E402
from src.utils.logging_config import setup_logging  # noqa: E402

logger = logging.getLogger(__name__)


def main() -> None:
    """Configure logging and launch the chess GUI."""
    setup_logging(level=logging.INFO)
    logger.info("Launching ChessAI...")

    gui = ChessGUI(ai_color=AI_COLOR, search_depth=SEARCH_DEPTH)
    gui.run()

    logger.info("ChessAI closed.")


if __name__ == "__main__":
    main()
