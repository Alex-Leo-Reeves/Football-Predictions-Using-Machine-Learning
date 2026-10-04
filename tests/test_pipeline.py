"""End-to-end pipeline tests (lightweight backend, small synthetic data)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.synthetic import SyntheticLeague
from src.features.builder import FeatureBuilder
from src.models.master_brain import MasterBrain
from src.pipeline.shadow import run_shadow_test
from src.pipeline.training import train_engine


def test_train_engine():
    league = SyntheticLeague(n_teams=8, n_seasons=2, matches_per_season=14)
    engine = train_engine(league.history, backend="hist")
    assert engine.goal_worker is not None
    assert engine.metrics["n_matches"] > 0


def test_master_brain_evaluates_fixture():
    league = SyntheticLeague(n_teams=8, n_seasons=2, matches_per_season=14)
    engine = train_engine(league.history, backend="hist")
    builder = FeatureBuilder(league.history)
    brain = MasterBrain(goal_worker=engine.goal_worker, anomaly_auditor=engine.anomaly_auditor)
    fixture = league.today_fixtures(1)[0]
    feats = builder.fixture_features(fixture)
    decision = brain.evaluate_fixture(fixture, feats)
    assert decision["decision"] in ("EXECUTE", "SKIP")
    assert "fixture" in decision


def test_shadow_backtest_runs():
    league = SyntheticLeague(n_teams=8, n_seasons=2, matches_per_season=14)
    engine = train_engine(league.history, backend="hist")
    report = run_shadow_test(league.history, engine, n_splits=2, min_train_frac=0.6)
    assert report["matches_evaluated"] > 0
    assert 0.0 <= report["precision"] <= 1.0
    assert 0.0 <= report["abstention_rate"] <= 1.0
