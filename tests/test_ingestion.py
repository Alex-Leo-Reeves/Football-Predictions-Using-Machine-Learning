"""Tests for the real-data mapping helpers (mock JSON, no live keys)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.api_football import parse_fixture_basic, parse_statistics
from src.data.ingestion import build_real_history
from src.data.odds_api import parse_odds_payload


def test_parse_fixture_basic():
    item = {
        "fixture": {"id": 123, "date": "2026-10-04T15:00:00+00:00",
                    "status": {"short": "FT"}},
        "league": {"id": 39, "name": "Premier League", "season": 2025},
        "teams": {"home": {"id": 40, "name": "Liverpool"},
                  "away": {"id": 50, "name": "Everton"}},
        "goals": {"home": 2, "away": 1},
    }
    b = parse_fixture_basic(item)
    assert b["fixture_id"] == 123
    assert b["home_name"] == "Liverpool"
    assert b["home_goals"] == 2
    assert b["status"] == "FT"


def test_parse_statistics():
    stats = [
        {"team": {"id": 40}, "statistics": [
            {"type": "expected_goals", "value": "1.8"},
            {"type": "Corner Kicks", "value": 7},
            {"type": "Yellow Cards", "value": 2},
            {"type": "Ball Possession", "value": "60%"},
        ]},
        {"team": {"id": 50}, "statistics": [
            {"type": "expected_goals", "value": "0.6"},
            {"type": "Corner Kicks", "value": 3},
            {"type": "Yellow Cards", "value": 4},
        ]},
    ]
    out = parse_statistics(stats)
    assert out["home"]["expected_goals"] == "1.8"
    assert out["home"]["Corner Kicks"] == 7
    assert out["away"]["Yellow Cards"] == 4
    # non-whitelisted stat is dropped
    assert "Ball Possession" not in out["home"]


def test_parse_odds_payload():
    items = [{
        "home_team": "Liverpool", "away_team": "Everton",
        "bookmakers": [{
            "key": "book1", "markets": [
                {"key": "h2h", "outcomes": [
                    {"name": "Liverpool", "price": 1.5},
                    {"name": "Draw", "price": 4.0},
                    {"name": "Everton", "price": 6.0},
                ]},
                {"key": "totals", "outcomes": [
                    {"name": "Over", "point": 2.5, "price": 1.9},
                    {"name": "Under", "point": 2.5, "price": 1.9},
                ]},
            ],
        }],
    }]
    out = parse_odds_payload(items)
    key = "liverpool @ everton"
    assert key in out
    assert out[key]["home_win"] == 1.5
    assert out[key]["away_win"] == 6.0
    assert out[key]["over_2.5"] == 1.9


class _MockFootballClient:
    """Minimal fake exposing fixtures_by_season + fixture_statistics."""

    def __init__(self, by_season):
        self.by_season = by_season
        self.calls = []

    def fixtures_by_season(self, league, season):
        self.calls.append(("fixtures", season))
        return self.by_season.get(season, [])

    def fixture_statistics(self, fixture_id):
        self.calls.append(("stats", fixture_id))
        return []


def _fixture_item(fid, season, date, hg, ag):
    return {
        "fixture": {"id": fid, "date": date, "status": {"short": "FT"}},
        "league": {"id": 39, "name": "Premier League", "season": season},
        "teams": {"home": {"id": 40, "name": "Liverpool"},
                  "away": {"id": 50, "name": "Everton"}},
        "goals": {"home": hg, "away": ag},
    }


def test_build_real_history_multiple_seasons():
    client = _MockFootballClient({
        2022: [_fixture_item(1, 2022, "2022-08-06T15:00:00+00:00", 2, 1)],
        2023: [_fixture_item(2, 2023, "2023-08-12T15:00:00+00:00", 1, 1)],
        2024: [_fixture_item(3, 2024, "2024-08-17T15:00:00+00:00", 0, 2)],
    })
    results = build_real_history(client, 39, [2022, 2023, 2024], max_stats_fixtures=60)
    # all three seasons are merged into one chronological history
    assert len(results) == 3
    assert [r.match_id for r in results] == ["1", "2", "3"]
    assert results[0].home_goals == 2 and results[2].away_goals == 2
    # one fixtures request per season
    assert client.calls.count(("fixtures", 2022)) == 1
    assert client.calls.count(("fixtures", 2024)) == 1


def test_build_real_history_accepts_single_int():
    client = _MockFootballClient({2024: [_fixture_item(3, 2024, "2024-08-17T15:00:00+00:00", 0, 2)]})
    results = build_real_history(client, 39, 2024, max_stats_fixtures=60)
    assert len(results) == 1
    assert results[0].match_id == "3"
