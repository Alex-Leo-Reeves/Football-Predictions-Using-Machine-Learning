"""Tests for the OddsPapi parser + key rotation."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.oddspapi import parse_oddspapi_odds
from src.utils.key_rotation import KeyPool, parse_keys


def test_parse_oddspapi_odds():
    items = [{
        "fixtureId": "id123",
        "bookmakerOdds": {
            "pinnacle": {
                "markets": {
                    "101": {"outcomes": {
                        "101": {"players": {"0": {"price": 1.5}}},
                        "102": {"players": {"0": {"price": 4.0}}},
                        "103": {"players": {"0": {"price": 6.0}}},
                    }},
                    "104": {"outcomes": {
                        "104": {"players": {"0": {"price": 1.9}}},
                        "105": {"players": {"0": {"price": 1.9}}},
                    }},
                }
            }
        },
    }]
    # market definitions from GET /v4/markets
    market_defs = {
        "101": {"marketType": "1x2", "outcomes": {"101": "1", "102": "X", "103": "2"}},
        "104": {"marketType": "totals", "outcomes": {"104": "Over", "105": "Under"}, "handicap": 2.5},
    }
    out = parse_oddspapi_odds(items, market_defs)
    assert "id123" in out
    assert out["id123"]["home_win"] == 1.5
    assert out["id123"]["draw"] == 4.0
    assert out["id123"]["over_2.5"] == 1.9
    assert out["id123"]["under_2.5"] == 1.9


def test_key_pool_rotation():
    pool = KeyPool(["k1", "k2"], budget_per_key=2, state_file=None)
    # exhaust k1
    assert pool.acquire() == "k1"
    pool.record_use("k1")
    assert pool.acquire() == "k2"
    pool.record_use("k2")
    assert pool.acquire() == "k1"
    pool.record_use("k1")
    # k1 now exhausted -> next acquire returns k2
    assert pool.acquire() == "k2"
    pool.record_use("k2")
    # both exhausted
    assert pool.acquire() is None
    assert pool.requests_left == 0


def test_parse_keys_comma_separated(monkeypatch):
    monkeypatch.setenv("FOOTBALL_API_KEYS", "a,b,c")
    monkeypatch.setenv("FOOTBALL_API_KEY", "a")
    keys = parse_keys("FOOTBALL_API_KEYS", "FOOTBALL_API_KEY")
    assert keys == ["a", "b", "c"]
