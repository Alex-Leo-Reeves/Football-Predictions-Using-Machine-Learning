#!/usr/bin/env python3
"""Forward-only shadow backtest. Usage: python scripts/backtest.py"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import get, load_config
from src.data.ingestion import DataLoader
from src.pipeline.shadow import run_shadow_test
from src.pipeline.training import train_engine


def main() -> None:
    config = load_config()
    loader = DataLoader()
    history = loader.load_history()
    backend = str(get("models.backend", "auto", config))
    engine = train_engine(history, backend=backend)
    report = run_shadow_test(history, engine)
    print(json.dumps({k: v for k, v in report.items() if k != "results"}, indent=2))
    print(f"\nPrecision (executed picks that hit): {report['precision']:.2%}")
    print(f"Abstention rate: {report['abstention_rate']:.2%}")


if __name__ == "__main__":
    main()
