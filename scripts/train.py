#!/usr/bin/env python3
"""Train the engine and print metrics. Usage: python scripts/train.py"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import get, load_config
from src.data.ingestion import DataLoader
from src.pipeline.training import train_engine


def main() -> None:
    config = load_config()
    loader = DataLoader()
    history = loader.load_history()
    backend = str(get("models.backend", "auto", config))
    engine = train_engine(history, backend=backend)
    print(json.dumps(engine.to_dict(), indent=2))
    print(f"Trained on {len(history)} matches using backend '{engine.backend}'")


if __name__ == "__main__":
    main()
