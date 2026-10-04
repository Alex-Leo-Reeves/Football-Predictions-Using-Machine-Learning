"""Probability calibration (isotonic / sigmoid).

Raw gradient-boosted probabilities are overconfident. This layer fits a
calibrator on out-of-fold predictions so that a predicted probability
matches the real-world outcome frequency.
"""
from __future__ import annotations

from typing import Any, List, Optional

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression

from src.config import get


class ProbabilityCalibrator:
    """Wraps a base classifier and calibrates its probabilities."""

    def __init__(self, base_model: Any, method: str | None = None, cv: int | None = None):
        self.base_model = base_model
        self.method = method or str(get("models.calibration.method", "isotonic"))
        self.cv = cv or int(get("models.calibration.cv", 5))
        self.calibrated: Any = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ProbabilityCalibrator":
        self.calibrated = CalibratedClassifierCV(
            estimator=self.base_model,
            method=self.method,
            cv=self.cv,
        )
        self.calibrated.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.calibrated is None:
            raise RuntimeError("Calibrator not fitted")
        return self.calibrated.predict_proba(X)


def fit_isotonic(scores: np.ndarray, labels: np.ndarray) -> IsotonicRegression:
    """Fit an isotonic regressor mapping raw scores -> calibrated probs."""
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(scores, labels)
    return iso


def calibrate_scores(iso: IsotonicRegression, scores: np.ndarray) -> np.ndarray:
    """Apply a fitted isotonic regressor."""
    return iso.predict(scores)
