"""Training pipeline.

Trains the full worker ensemble + auditors on leakage-safe features and
returns a ready-to-use engine bundle. Heavy training runs on GitHub
Actions; locally it uses the lightweight HistGradientBoosting backend.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.config import get
from src.data.schemas import MatchResult
from src.features.builder import FEATURE_COLUMNS, FeatureBuilder
from src.models.anomaly import AnomalyAuditor
from src.models.base import make_classifier, resolve_backend
from src.models.conformal import ConformalClassifier
from src.models.workers import BasketballWorker, CountWorker, GoalWorker


class TrainedEngine:
    """Bundle of trained workers + auditors + metadata."""

    def __init__(self, goal_worker: GoalWorker, corner_worker: CountWorker | None,
                 card_worker: CountWorker | None, anomaly_auditor: AnomalyAuditor | None,
                 conformal: ConformalClassifier | None, backend: str,
                 feature_columns: List[str], metrics: Dict[str, Any] | None = None):
        self.goal_worker = goal_worker
        self.corner_worker = corner_worker
        self.card_worker = card_worker
        self.anomaly_auditor = anomaly_auditor
        self.conformal = conformal
        self.backend = backend
        self.feature_columns = feature_columns
        self.metrics = metrics or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "backend": self.backend,
            "feature_columns": self.feature_columns,
            "metrics": self.metrics,
        }


def train_engine(history: List[MatchResult], backend: str = "auto",
                 train_conformal: bool = True) -> TrainedEngine:
    """Train the full engine on historical matches."""
    backend = resolve_backend(backend)
    builder = FeatureBuilder(history)
    X, y = builder.training_matrix(history)

    # ---- Goal worker ----
    goal_worker = GoalWorker(backend=backend).fit(X, y)

    # ---- Corner & card workers ----
    corner_worker = None
    card_worker = None
    if "home_corners" in y.columns and y["home_corners"].sum() > 0:
        corner_worker = CountWorker("home_corners", backend=backend).fit(X, y)
    if "home_cards" in y.columns and y["home_cards"].sum() > 0:
        card_worker = CountWorker("home_cards", backend=backend).fit(X, y)

    # ---- Anomaly auditor ----
    anomaly_auditor = AnomalyAuditor().fit(X.values)

    # ---- Conformal classifier (outcome) ----
    conformal = None
    if train_conformal:
        clf = make_classifier(backend)
        conformal = ConformalClassifier(clf).fit(X.values, y["outcome"].values)

    # ---- Quick metrics ----
    metrics = _quick_metrics(goal_worker, X, y)

    return TrainedEngine(
        goal_worker=goal_worker,
        corner_worker=corner_worker,
        card_worker=card_worker,
        anomaly_auditor=anomaly_auditor,
        conformal=conformal,
        backend=backend,
        feature_columns=FEATURE_COLUMNS,
        metrics=metrics,
    )


def _quick_metrics(goal_worker: GoalWorker, X: pd.DataFrame, y: pd.DataFrame) -> Dict[str, Any]:
    """In-sample MAE for the goal workers (diagnostic only)."""
    lambdas = goal_worker.predict_lambdas(X)
    mae_home = float(np.mean(np.abs(lambdas["home_lambda"] - y["home_goals"].values)))
    mae_away = float(np.mean(np.abs(lambdas["away_lambda"] - y["away_goals"].values)))
    return {
        "mae_home_goals": round(mae_home, 4),
        "mae_away_goals": round(mae_away, 4),
        "n_matches": int(len(X)),
        "backend": goal_worker.backend,
    }


def train_basketball_engine(history: List[MatchResult], backend: str = "auto") -> TrainedEngine:
    """Train a basketball-only engine (points worker + anomaly auditor).

    Uses the same leakage-safe feature matrix as football but targets
    POINTS. Returns a :class:`TrainedEngine` whose ``goal_worker`` slot is a
    :class:`BasketballWorker` (the pipeline branches on ``sport``).
    """
    backend = resolve_backend(backend)
    builder = FeatureBuilder(history)
    X, y = builder.training_matrix(history)

    basketball_worker = BasketballWorker(backend=backend).fit(X, y)
    anomaly_auditor = AnomalyAuditor().fit(X.values)

    metrics = {
        "mae_home_points": round(float(np.mean(np.abs(
            basketball_worker.predict_points(X)["home_lambda"] - y["home_goals"].values))), 4),
        "mae_away_points": round(float(np.mean(np.abs(
            basketball_worker.predict_points(X)["away_lambda"] - y["away_goals"].values))), 4),
        "n_matches": int(len(X)),
        "backend": backend,
        "sport": "basketball",
    }

    return TrainedEngine(
        goal_worker=basketball_worker,  # slot reused; sport discriminates
        corner_worker=None,
        card_worker=None,
        anomaly_auditor=anomaly_auditor,
        conformal=None,
        backend=backend,
        feature_columns=FEATURE_COLUMNS,
        metrics=metrics,
    )
