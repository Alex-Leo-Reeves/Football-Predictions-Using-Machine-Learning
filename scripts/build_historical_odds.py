#!/usr/bin/env python3
"""Build a historical-odds training dataset (edge model).

OddsPapi provides FREE historical odds with line-movement. This script:
  1. loads completed fixtures from API-Football
  2. fetches OddsPapi historical odds for each (budget-aware)
  3. saves {fixture, closing odds, actual result} to data/processed/
     for training the edge/EV model

Usage:
  export FOOTBALL_API_KEY=... ODDSPAPI_KEY=... \
         python3 scripts/build_historical_odds.py --limit 20
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import get
from src.data.api_football import APIFootballClient, parse_fixture_basic
from src.data.oddspapi import OddsPapiClient, parse_oddspapi_odds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=20, help="max fixtures to process")
    parser.add_argument("--bookmaker", default="pinnacle")
    args = parser.parse_args()

    league = int(get("data.football_api.league_id", 39))
    seasons = get("data.football_api.seasons", [int(get("data.football_api.season", 2024))])
    if isinstance(seasons, int):
        seasons = [seasons]

    fb = APIFootballClient()
    op = OddsPapiClient()

    fixtures = []
    for season in seasons:
        fixtures.extend(fb.fixtures_by_season(league, season))
    completed = [f for f in fixtures if parse_fixture_basic(f).get("status") in ("FT", "AET", "PEN")]
    completed.sort(key=lambda f: parse_fixture_basic(f).get("kickoff") or "")
    completed = completed[-args.limit:]
    print(f"Processing {len(completed)} completed fixtures across seasons {seasons}")

    rows = []
    for item in completed:
        b = parse_fixture_basic(item)
        fid = b.get("fixture_id")
        if fid is None:
            continue
        hist = op.historical_odds(str(fid), bookmaker=args.bookmaker)
        parsed = parse_oddspapi_odds(hist if isinstance(hist, list) else [])
        odds = parsed.get(str(fid), {})
        rows.append({
            "fixture_id": fid,
            "home": b["home_name"], "away": b["away_name"],
            "kickoff": b["kickoff"],
            "home_goals": b["home_goals"], "away_goals": b["away_goals"],
            "outcome": "H" if b["home_goals"] > b["away_goals"] else ("D" if b["home_goals"] == b["away_goals"] else "A"),
            "closing_odds": odds,
        })

    out_dir = Path(get("data.cache_dir", "data/raw")).parent / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "historical_odds.json"
    out_file.write_text(json.dumps(rows, indent=2, default=str))
    print(f"Saved {len(rows)} rows to {out_file}")
    print(f"Requests left today: {op.requests_left}")


if __name__ == "__main__":
    main()
