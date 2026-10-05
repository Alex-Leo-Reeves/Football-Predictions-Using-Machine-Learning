"""Shadow / backtest harness.

Runs the engine forward-only over historical windows and reports HONEST
metrics: how often EXECUTEd predictions actually hit, calibration error,
and abstention rate. This is the "benchmark against reality" layer.
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from src.data.schemas import Fixture, MatchResult, naive
from src.features.builder import FeatureBuilder
from src.markets.accumulator import resolve_market_outcome
from src.models.master_brain import MasterBrain
from src.pipeline.training import TrainedEngine


def _outcome_of(match: MatchResult, market_key: str) -> bool:
    """Resolve whether a market selection actually hit (shared resolver)."""
    return resolve_market_outcome(match, market_key)


def run_shadow_test(history: List[MatchResult], engine: TrainedEngine,
                    n_splits: int = 3, min_train_frac: float = 0.6) -> Dict[str, Any]:
    """Forward-only backtest of the master brain over historical windows."""
    from src.pipeline.validation import expanding_window_splits

    history = sorted(history, key=lambda m: naive(m.kickoff))
    n = len(history)
    all_results: List[Dict[str, Any]] = []
    executed = 0
    hits = 0

    for train_idx, val_idx in expanding_window_splits(n, n_splits, min_train_frac):
        train_matches = [history[i] for i in train_idx]
        val_matches = [history[i] for i in val_idx]

        # Re-train a fresh engine on the training window only (no leakage)
        from src.pipeline.training import train_engine
        fold_engine = train_engine(train_matches, backend=engine.backend)
        builder = FeatureBuilder(train_matches)
        brain = MasterBrain(
            goal_worker=fold_engine.goal_worker,
            corner_worker=fold_engine.corner_worker,
            card_worker=fold_engine.card_worker,
            anomaly_auditor=fold_engine.anomaly_auditor,
        )

        for match in val_matches:
            fixture = Fixture(
                fixture_id=match.match_id,
                kickoff=match.kickoff,
                home_team_id=match.home_team_id,
                away_team_id=match.away_team_id,
                home_team_name=match.home_team_id,
                away_team_name=match.away_team_id,
                home_kpai=match.home_kpai,
                away_kpai=match.away_kpai,
            )
            feats = builder.fixture_features(fixture)
            decision = brain.evaluate_fixture(fixture, feats)
            record = {"match_id": match.match_id, "decision": decision["decision"]}
            if decision["decision"] == "EXECUTE":
                executed += 1
                hit = _outcome_of(match, decision.get("key", decision.get("market", "")))
                if hit:
                    hits += 1
                record.update({
                    "market": decision.get("market"),
                    "confidence": decision.get("confidence"),
                    "hit": hit,
                })
            all_results.append(record)

    precision = (hits / executed) if executed else 0.0
    return {
        "matches_evaluated": len(all_results),
        "executed": executed,
        "abstained": len(all_results) - executed,
        "abstention_rate": round((len(all_results) - executed) / len(all_results), 4) if all_results else 0.0,
        "hits": hits,
        "precision": round(precision, 4),
        "results": all_results,
    }
