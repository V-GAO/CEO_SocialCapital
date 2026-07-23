"""Standardized logging setup for the CEO_SC pipeline."""

from __future__ import annotations

import logging
import sys

_CONFIGURED = False


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logging once with a consistent format."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    setup_logging()
    return logging.getLogger(name)
