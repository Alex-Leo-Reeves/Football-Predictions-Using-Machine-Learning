"""Lightweight logging helpers."""
from __future__ import annotations

import logging
import sys

_LOGGER: logging.Logger | None = None


def get_logger(name: str = "football-engine") -> logging.Logger:
    global _LOGGER
    if _LOGGER is None:
        _LOGGER = logging.getLogger(name)
        _LOGGER.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S"))
        _LOGGER.addHandler(handler)
    return _LOGGER
