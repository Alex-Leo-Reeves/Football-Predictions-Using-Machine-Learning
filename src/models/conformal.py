"""Split conformal prediction (implemented from scratch).

Conformal prediction gives a *guaranteed* coverage bound: with
probability >= 1-alpha, the true label is inside the predicted set.
When the set contains more than one class, the engine abstains (SKIP).

This is the mathematical backbone of the "only predict when certain"
requirement — implemented dependency-free so it runs anywhere.
"""
from __future__ import annotations

from typing import Any, List, Sequence, Tuple

import numpy as np

from src.config import get


class ConformalClassifier:
    """Split conformal wrapper around any sklearn-like classifier."""

    def __init__(self, base_model: Any, alpha: float | None = None,
                 calibration_fraction: float | None = None, random_state: int = 42):
        self.base_model = base_model
        self.alpha = alpha if alpha is not None else float(get("models.conformal.alpha", 0.10))
        self.calibration_fraction = calibration_fraction if calibration_fraction is not None else float(
            get("models.conformal.calibration_fraction", 0.20))
        self.random_state = random_state
        self.classes_: np.ndarray | None = None
        self.qhat_: float = 0.0
        self.cal_scores_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ConformalClassifier":
        """Split data, fit base model, compute calibration quantile.

        String labels (e.g. ``['A','D','H']``) are encoded to integers before
        fitting — XGBoost requires numeric classes — and mapped back to the
        original labels in :meth:`predict_set`.
        """
        rng = np.random.default_rng(self.random_state)
        n = len(X)
        n_cal = max(10, int(n * self.calibration_fraction))
        idx = rng.permutation(n)
        cal_idx, train_idx = idx[:n_cal], idx[n_cal:]

        X_train, y_train = X[train_idx], y[train_idx]
        X_cal, y_cal = X[cal_idx], y[cal_idx]

        # Encode string labels to integers (XGBoost needs numeric classes).
        self.label_map_: dict | None = None
        if y.dtype.kind in "OUS" or any(isinstance(v, str) for v in y[:10]):
            unique = np.unique(y)
            self.label_map_ = {label: i for i, label in enumerate(unique)}
            y_train = np.array([self.label_map_[v] for v in y_train])
            y_cal = np.array([self.label_map_[v] for v in y_cal])

        self.base_model.fit(X_train, y_train)
        self.classes_ = np.asarray(self.base_model.classes_)

        # Conformity scores on calibration set: 1 - prob of true class
        probs = self.base_model.predict_proba(X_cal)
        true_probs = probs[np.arange(len(y_cal)), np.searchsorted(self.classes_, y_cal)]
        scores = 1.0 - true_probs
        self.cal_scores_ = scores

        # Quantile with finite-sample correction
        q_level = np.ceil((len(scores) + 1) * (1.0 - self.alpha)) / len(scores)
        q_level = min(1.0, q_level)
        self.qhat_ = float(np.quantile(scores, q_level, method="higher"))
        return self

    def predict_set(self, X: np.ndarray) -> List[List[str]]:
        """Return the prediction set (list of class labels) per row."""
        if self.classes_ is None:
            raise RuntimeError("ConformalClassifier not fitted")
        probs = self.base_model.predict_proba(X)
        inv_map = None
        if self.label_map_ is not None:
            inv_map = {v: k for k, v in self.label_map_.items()}
        sets: List[List[str]] = []
        for row in probs:
            included = [int(c) for c, p in zip(self.classes_, row) if p >= 1.0 - self.qhat_]
            if inv_map is not None:
                included = [inv_map[c] for c in included]
            sets.append([str(c) for c in included])
        return sets

    def predict_single(self, x: np.ndarray) -> Tuple[List[str], float]:
        """Prediction set + max probability for a single row."""
        sets = self.predict_set(x.reshape(1, -1))
        probs = self.base_model.predict_proba(x.reshape(1, -1))[0]
        return sets[0], float(probs.max())

    def coverage(self, X: np.ndarray, y: np.ndarray) -> float:
        """Empirical coverage on a labeled set (should be >= 1-alpha)."""
        sets = self.predict_set(X)
        hits = 0
        for s, true in zip(sets, y):
            if str(true) in s:
                hits += 1
        return hits / len(y) if len(y) else 0.0
