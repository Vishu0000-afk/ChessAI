"""Standard logging configuration for the ChessAI project."""

from __future__ import annotations

import logging
import sys


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logging for the application.

    Sets a concise console format. Called once from the application
    entry point (main.py). Library modules should just use
    `logging.getLogger(__name__)` and never configure handlers themselves.

    Args:
        level: Logging level (e.g. logging.INFO, logging.DEBUG).
    """
    root_logger = logging.getLogger()

    if root_logger.handlers:
        # Already configured (e.g. in tests); avoid duplicate handlers.
        root_logger.setLevel(level)
        return

    handler = logging.StreamHandler(stream=sys.stdout)
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    handler.setFormatter(formatter)

    root_logger.addHandler(handler)
    root_logger.setLevel(level)

    # Keep noisy third-party libraries quiet unless debugging.
    logging.getLogger("PIL").setLevel(logging.WARNING)
