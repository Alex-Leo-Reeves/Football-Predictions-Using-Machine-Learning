"""Anomaly / variance auditor.

Flags fixtures that live in an unusual region of feature space
(out-of-distribution) — e.g. a brand-new manager, extreme weather proxy,
or wildly inconsistent form. If flagged, the master brain applies a
penalty or hard-skips the fixture.
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
from sklearn.ensemble import IsolationForest

from src.config import get


class AnomalyAuditor:
    """IsolationForest-based auditor that vetoes out-of-distribution fixtures."""

    def __init__(self, contamination: float | None = None, random_state: int = 42):
        self.contamination = contamination if contamination is not None else float(
            get("models.anomaly.contamination", 0.05))
        self.random_state = random_state
        self.model: IsolationForest | None = None

    def fit(self, X: np.ndarray) -> "AnomalyAuditor":
        self.model = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.model.fit(X)
        return self

    def is_outlier(self, x: np.ndarray) -> bool:
        """True if the fixture is flagged as an anomaly."""
        if self.model is None:
            return False
        pred = self.model.predict(x.reshape(1, -1))
        return bool(pred[0] == -1)

    def anomaly_score(self, x: np.ndarray) -> float:
        if self.model is None:
            return 0.0
        return float(self.model.score_samples(x.reshape(1, -1))[0])


class VarianceAuditor:
    """Rejects fixtures where a team's recent xG is wildly inconsistent."""

    def __init__(self, max_std: float = 1.2):
        self.max_std = max_std

    def check(self, feats: Dict[str, float]) -> List[str]:
        flags: List[str] = []
        for key, label in (("xgf_std_10_home", "home"), ("xgf_std_10_away", "away")):
            if feats.get(key, 0.0) > self.max_std:
                flags.append(f"high_variance_{label}")
        return flags
