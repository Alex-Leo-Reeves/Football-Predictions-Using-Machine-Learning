#!/usr/bin/env python3
"""Run the daily prediction pipeline. Usage: python scripts/run_daily.py"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.main_pipeline import run_daily_pipeline


def main() -> None:
    config = load_config()
    result = run_daily_pipeline(config)
    print(json.dumps(result["ticket"], indent=2, default=str))


if __name__ == "__main__":
    main()
