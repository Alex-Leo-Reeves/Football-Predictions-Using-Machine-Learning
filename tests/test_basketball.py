"""Tests for basketball support and time-window filtering."""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.ingestion import filter_fixtures_by_time
from src.data.schemas import Fixture
from src.data.synthetic_basketball import SyntheticBasketballLeague
from src.models.basketball import basketball_probabilities
from src.models.workers import BasketballWorker
from src.pipeline.training import train_basketball_engine


def test_basketball_probabilities_sum_to_one():
    probs = basketball_probabilities(112.0, 108.0, spread=-5.5, total_line=220.5)
    assert abs(probs["home_win"] + probs["away_win"] - 1.0) < 1e-9
    assert abs(probs["home_cover"] + probs["away_cover"] - 1.0) < 1e-9
    assert abs(probs["over"] + probs["under"] - 1.0) < 1e-9
    # a 4-point favourite should be more likely to win than lose
    assert probs["home_win"] > 0.5


def test_basketball_probabilities_favourite_covers():
    # home favoured by 5.5 with a big expected edge -> covers more often
    probs = basketball_probabilities(118.0, 100.0, spread=-5.5)
    assert probs["home_cover"] > 0.5
    assert probs["home_win"] > 0.8


def test_synthetic_basketball_history():
    league = SyntheticBasketballLeague(n_teams=8, n_seasons=2, matches_per_season=10)
    assert len(league.history) == 20
    for m in league.history:
        assert m.sport == "basketball"
        # basketball is high-scoring
        assert m.home_goals > 60 and m.away_goals > 60


def test_synthetic_basketball_fixtures_have_sport():
    league = SyntheticBasketballLeague(n_teams=8, n_seasons=1, matches_per_season=5)
    fixtures = league.today_fixtures(5)
    assert len(fixtures) == 5
    assert all(f.sport == "basketball" for f in fixtures)


def test_train_basketball_engine():
    league = SyntheticBasketballLeague(n_teams=8, n_seasons=2, matches_per_season=10)
    engine = train_basketball_engine(league.history, backend="hist")
    assert engine.metrics["sport"] == "basketball"
    assert engine.metrics["n_matches"] > 0
    assert isinstance(engine.goal_worker, BasketballWorker)


def _fixture(fid: str, kickoff: datetime) -> Fixture:
    return Fixture(
        fixture_id=fid,
        kickoff=kickoff,
        home_team_id="H",
        away_team_id="A",
        home_team_name="Home",
        away_team_name="Away",
    )


def test_time_filter_excludes_past_and_after_window():
    now = datetime(2026, 10, 4, 10, 0, 0, tzinfo=timezone.utc)
    fixtures = [
        _fixture("past", now - timedelta(hours=1)),    # 9am -> already started
        _fixture("now", now + timedelta(minutes=5)),   # 10:05 -> keep
        _fixture("mid", now + timedelta(hours=5)),     # 15:00 -> keep
        _fixture("late", now.replace(hour=23, minute=45)),  # 23:45 -> after window
    ]
    kept = filter_fixtures_by_time(fixtures, now=now, end_time="23:30")
    ids = {f.fixture_id for f in kept}
    assert "past" not in ids
    assert "now" in ids
    assert "mid" in ids
    assert "late" not in ids


def test_time_filter_handles_naive_datetimes():
    now = datetime(2026, 10, 4, 10, 0, 0)  # naive
    fixtures = [
        _fixture("past", datetime(2026, 10, 4, 9, 0, 0)),
        _fixture("future", datetime(2026, 10, 4, 12, 0, 0)),
    ]
    kept = filter_fixtures_by_time(fixtures, now=now, end_time="23:30")
    assert {f.fixture_id for f in kept} == {"future"}
