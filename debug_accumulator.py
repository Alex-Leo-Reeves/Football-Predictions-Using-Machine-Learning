"""Temporary debug script for the accumulator optimizer."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.markets.accumulator import AccumulatorBuilder

builder = AccumulatorBuilder(target_odds=300.0, min_confidence=0.8, target_tolerance=0.05)
selections = [
    {
        "fixture_id": f"F{i}", "home_id": f"H{i}", "away_id": f"A{i}",
        "market": "Home Win", "key": "home_win", "confidence": 0.85, "odds": 1.8,
        "variants": [
            {"market": "DC 1X", "key": "home_or_draw", "confidence": 0.92, "odds": 1.30},
            {"market": "DC 12", "key": "home_or_away", "confidence": 0.95, "odds": 1.10},
        ],
    }
    for i in range(25)
]
ticket = builder.build(selections)
print("legs:", ticket["total_legs"])
print("odds:", ticket["combined_odds"])
print("steps:", len(ticket["optimization_steps"]))
for s in ticket["optimization_steps"][:10]:
    print(s)
