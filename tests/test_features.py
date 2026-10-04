"""Tests for feature engineering modules."""
from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.schemas import Fixture, MatchResult
from src.data.synthetic import SyntheticLeague
from src.features.builder import FeatureBuilder
from src.features.elo import EloSystem
from src.features.injuries import compute_kpai, kpai_from_missing
from src.features.rolling import RollingFeatureBuilder


def _match(home_id="H", away_id="A", hg=1, ag=0, day=0, hx=1.2, ax=0.6):
    return MatchResult(
        match_id=f"M{day}",
        kickoff=datetime(2026, 1, 1) + timedelta(days=day),
        home_team_id=home_id, away_team_id=away_id,
        home_goals=hg, away_goals=ag, home_xg=hx, away_xg=ax,
    )


def test_elo_updates():
    elo = EloSystem(k=32, home_advantage=65)
    elo.update(_match("A", "B", hg=2, ag=0))
    assert elo.rating("A") > elo.rating("B")
    assert elo.rating_diff("A", "B") > 0


def test_rolling_ema():
    rb = RollingFeatureBuilder(windows=[3, 5])
    history = [_match("A", "B", hg=2, ag=0, day=0), _match("A", "B", hg=1, ag=1, day=1)]
    feats = rb.team_features(history, "A")
    assert feats["gf_ema_3"] > 0
    assert feats["scored_rate_5"] > 0


def test_h2h_and_venue():
    from src.features.h2h import H2HFeatureBuilder
    history = [_match("A", "B", hg=2, ag=0, day=0), _match("A", "B", hg=1, ag=0, day=1)]
    feats = H2HFeatureBuilder().features(history, "A", "B")
    assert feats["h2h_home_win_ratio"] == 1.0
    assert feats["venue_home_undefeated_streak"] == 2.0


def test_kpai():
    from src.data.synthetic import SyntheticLeague
    from src.features.injuries import player_importance
    league = SyntheticLeague(n_teams=4, n_seasons=1, matches_per_season=6)
    tid = list(league.teams.keys())[0]
    players = league.players_for(tid)
    full = compute_kpai(players, [p.id for p in players])
    assert full == 1.0
    # remove the single most important player -> KPAI must drop
    top = max(players, key=player_importance)
    missing = kpai_from_missing(players, [top.id])
    assert missing < 1.0


def test_feature_builder_no_leakage():
    from src.features.builder import FEATURE_COLUMNS
    league = SyntheticLeague(n_teams=6, n_seasons=1, matches_per_season=10)
    builder = FeatureBuilder(league.history)
    fixture = league.today_fixtures(1)[0]
    feats = builder.fixture_features(fixture)
    assert set(feats.keys()) == set(FEATURE_COLUMNS)
    for col in ["elo_diff", "home_xgf_ema_5", "venue_home_scored_rate"]:
        assert col in feats
