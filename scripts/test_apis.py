#!/usr/bin/env python3
"""Test all API keys. Usage:
  export FOOTBALL_API_KEY=... ODDS_API_KEY=... ODDSPAPI_KEY=... \
         python3 scripts/test_apis.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import get
from src.data.api_football import APIFootballClient
from src.data.odds_api import OddsAPIClient
from src.data.oddspapi import OddsPapiClient


def main() -> None:
    league = int(get("data.football_api.league_id", 39))
    seasons = get("data.football_api.seasons", [int(get("data.football_api.season", 2024))])
    if isinstance(seasons, int):
        seasons = [seasons]

    print("== API-Football ==")
    try:
        fb = APIFootballClient()
        for season in seasons:
            fixtures = fb.fixtures_by_season(league, season)
            print(f"  league {league} season {season}: {len(fixtures)} fixtures")
            if fixtures:
                from src.data.api_football import parse_fixture_basic
                b = parse_fixture_basic(fixtures[0])
                print(f"  sample: {b['home_name']} {b['home_goals']}-{b['away_goals']} {b['away_name']} ({b['status']})")
        print(f"  requests left today: {fb.requests_left} (keys: {fb.n_keys})")
        print("  ✅ API-Football key works")
    except RuntimeError as exc:
        print(f"  ❌ {exc}")

    print("\n== The Odds API (fallback) ==")
    try:
        odds = OddsAPIClient()
        sport = get("data.odds_api.sport", "soccer_epl")
        items = odds.odds(sport, get("data.odds_api.regions", "eu"))
        print(f"  sport {sport}: {len(items)} matches with odds (keys: {odds.n_keys})")
        if items:
            print(f"  sample: {items[0].get('home_team')} vs {items[0].get('away_team')}")
        print("  ✅ Odds API key works")
    except RuntimeError as exc:
        print(f"  ❌ {exc}")

    print("\n== OddsPapi (primary) ==")
    try:
        op = OddsPapiClient()
        tournaments = op.tournaments(sport_id=10)
        print(f"  soccer tournaments: {len(tournaments)} (keys: {op.n_keys})")
        print(f"  requests left today: {op.requests_left}")
        if tournaments:
            print(f"  sample: {tournaments[0].get('tournamentName')}")
        print("  ✅ OddsPapi key works")
    except RuntimeError as exc:
        print(f"  ❌ {exc}")


if __name__ == "__main__":
    main()
