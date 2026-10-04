"""Tests for the market scanner and accumulator."""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.schemas import MatchResult
from src.markets.accumulator import (
    AccumulatorBuilder,
    resolve_market_outcome,
    settle_ticket,
)
from src.markets.registry import all_markets, market_count
from src.markets.scanner import UniversalMarketScanner
from src.models.poisson import match_probabilities


def test_market_registry():
    assert market_count() > 100
    assert "1X2: Home Win" in all_markets()
    assert "BTTS: Yes (GG)" in all_markets()


def test_scanner_picks_best():
    probs = match_probabilities(2.5, 0.6)
    scanner = UniversalMarketScanner(min_edge=0.0)
    result = scanner.scan("A vs B", probs)
    assert result["decision"] in ("EXECUTE", "SKIP")
    assert result["evaluated_markets"] > 0


def test_accumulator_independence():
    builder = AccumulatorBuilder(target_odds=10.0, min_confidence=0.8)
    selections = [
        {"fixture_id": "F1", "home_id": "A", "away_id": "B", "market": "m1",
         "confidence": 0.9, "odds": 1.5},
        {"fixture_id": "F1", "home_id": "A", "away_id": "B", "market": "m2",
         "confidence": 0.95, "odds": 1.4},  # same fixture -> must be rejected
        {"fixture_id": "F2", "home_id": "C", "away_id": "D", "market": "m3",
         "confidence": 0.9, "odds": 1.6},
    ]
    ticket = builder.build(selections)
    # m2 (same fixture, higher confidence 0.95) is picked; m1 rejected;
    # m3 (independent fixture) accepted -> 1.4 * 1.6 = 2.24
    assert ticket["total_legs"] == 2
    assert abs(ticket["combined_odds"] - 2.24) < 1e-6


def test_accumulator_target():
    # Over-select: 40 legs at 1.25 odds = 7523 combined. The optimizer
    # must bring it down toward 300 (within tolerance) using variants.
    builder = AccumulatorBuilder(target_odds=300.0, min_confidence=0.9)
    selections = [
        {
            "fixture_id": f"F{i}", "home_id": f"H{i}", "away_id": f"A{i}",
            "market": "m", "confidence": 0.92, "odds": 1.25,
            "variants": [
                {"market": "m2", "key": "k2", "confidence": 0.95, "odds": 1.10},
            ],
        }
        for i in range(40)
    ]
    ticket = builder.build(selections)
    # combined odds must land close to the 300 target (within tolerance)
    assert 285.0 <= ticket["combined_odds"] <= 315.0
    assert ticket["target_reached"] is True


def test_accumulator_overshoot_trimmed_to_target():
    """The user's use case: 25 picks = 450 odds -> trim/downgrade to ~300."""
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
    # 25 legs at 1.8 = 1.8^25 ~ 45,000 -> must be trimmed hard toward 300
    assert 285.0 <= ticket["combined_odds"] <= 315.0
    assert ticket["target_reached"] is True
    # the optimizer must have taken downgrade/removal steps
    assert ticket["optimization_steps"] != []


def test_accumulator_keeps_all_when_below_target():
    """Never force inaccurate picks: if the pool can't reach 300, keep all."""
    builder = AccumulatorBuilder(target_odds=300.0, min_confidence=0.9)
    selections = [
        {"fixture_id": f"F{i}", "home_id": f"H{i}", "away_id": f"A{i}",
         "market": "m", "confidence": 0.95, "odds": 1.10}
        for i in range(5)
    ]
    ticket = builder.build(selections)
    assert ticket["total_legs"] == 5
    assert ticket["combined_odds"] < 300.0
    assert ticket["target_reached"] is False
    assert ticket["optimization_steps"] == []


def test_accumulator_all_or_nothing_flag():
    builder = AccumulatorBuilder(target_odds=10.0, min_confidence=0.8)
    selections = [
        {"fixture_id": "F1", "home_id": "A", "away_id": "B", "market": "m1",
         "confidence": 0.9, "odds": 1.5},
        {"fixture_id": "F2", "home_id": "C", "away_id": "D", "market": "m2",
         "confidence": 0.9, "odds": 1.6},
    ]
    ticket = builder.build(selections)
    assert ticket["all_or_nothing"] is True
    # system bets are OFF by default (one wrong leg = whole ticket loss)
    assert ticket["system_bets"] == []


def test_accumulator_system_bets_opt_in():
    builder = AccumulatorBuilder(target_odds=10.0, min_confidence=0.8, system_bets=True)
    selections = [
        {"fixture_id": f"F{i}", "home_id": f"H{i}", "away_id": f"A{i}",
         "market": "m", "confidence": 0.9, "odds": 1.5}
        for i in range(4)
    ]
    ticket = builder.build(selections)
    assert ticket["system_bets"] != []


def _match(fid: str, hg: int, ag: int) -> MatchResult:
    return MatchResult(
        match_id=fid,
        kickoff=datetime(2026, 10, 4, 15, 0),
        home_team_id="H", away_team_id="A",
        home_goals=hg, away_goals=ag,
    )


def test_resolve_market_outcome():
    m = _match("F1", 2, 1)   # total goals = 3
    assert resolve_market_outcome(m, "home_win") is True
    assert resolve_market_outcome(m, "away_win") is False
    assert resolve_market_outcome(m, "draw") is False
    assert resolve_market_outcome(m, "home_or_draw") is True
    assert resolve_market_outcome(m, "over_2.5") is True    # 3 goals > 2.5
    assert resolve_market_outcome(m, "under_2.5") is False  # 3 goals not < 2.5
    assert resolve_market_outcome(m, "over_2.8") is True    # 3 goals > 2.8
    assert resolve_market_outcome(m, "btts_yes") is True
    assert resolve_market_outcome(m, "home_clean_sheet") is False


def test_settle_ticket_all_or_nothing():
    results = {
        "F1": _match("F1", 2, 1),   # home_win hit
        "F2": _match("F2", 0, 1),   # home_win MISSED
    }
    ticket = {
        "selections": [
            {"fixture_id": "F1", "market": "1X2: Home Win", "key": "home_win"},
            {"fixture_id": "F2", "market": "1X2: Home Win", "key": "home_win"},
        ]
    }
    settled = settle_ticket(ticket, results)
    # one wrong leg -> the WHOLE ticket is a loss
    assert settled["status"] == "LOSS"
    assert settled["all_or_nothing"] is True
    assert settled["hit_legs"] == 1
    assert settled["total_legs"] == 2


def test_settle_ticket_win_only_when_all_hit():
    results = {
        "F1": _match("F1", 2, 1),
        "F2": _match("F2", 3, 0),
    }
    ticket = {
        "selections": [
            {"fixture_id": "F1", "key": "home_win"},
            {"fixture_id": "F2", "key": "over_2.5"},
        ]
    }
    settled = settle_ticket(ticket, results)
    assert settled["status"] == "WIN"
    assert settled["hit_legs"] == 2
