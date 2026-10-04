#!/usr/bin/env python3
"""Quick verification of the rich LLM context builder."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.ingestion import DataLoader
from src.main_pipeline import _build_llm_context

loader = DataLoader()
history = loader.load_history()
fixture = loader.load_today_fixtures()[0]
ctx = _build_llm_context(history, fixture, loader.players())
print("recent home:", len(ctx["recent_matches"]["home"]))
print("recent away:", len(ctx["recent_matches"]["away"]))
print("h2h meetings:", len(ctx["h2h"]))
print("injuries:", ctx["injuries"])
print("OK")
