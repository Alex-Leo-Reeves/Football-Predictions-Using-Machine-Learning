"""Worker models — specialized domain experts.

Each worker is trained on a single mathematical dimension of the match:
  * GoalWorker  -> expected home & away goals (lambda) via regressors
  * CornerWorker-> expected total corners
  * CardWorker  -> expected total cards

The GoalWorker's lambdas feed the Dixon-Coles Poisson matrix to derive
all goal-market probabilities.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.models.base import make_regressor
from src.models.poisson import match_probabilities


class GoalWorker:
    """Predicts expected goals for home and away teams."""

    def __init__(self, backend: str = "auto", **kwargs: Any):
        self.backend = backend
        self.model_home: Any = None
        self.model_away: Any = None
        self.feature_columns: List[str] = []

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> "GoalWorker":
        self.feature_columns = list(X.columns)
        self.model_home = make_regressor(self.backend)
        self.model_away = make_regressor(self.backend)
        self.model_home.fit(X, y["home_goals"].values)
        self.model_away.fit(X, y["away_goals"].values)
        return self

    def predict_lambdas(self, X: pd.DataFrame) -> Dict[str, float]:
        """Return predicted expected goals (lambdas) for each row."""
        X = X[self.feature_columns]
        home_lambda = np.clip(self.model_home.predict(X), 0.05, 6.0)
        away_lambda = np.clip(self.model_away.predict(X), 0.05, 6.0)
        return {"home_lambda": home_lambda, "away_lambda": away_lambda}

    def predict_probs(self, X: pd.DataFrame) -> List[Dict[str, float]]:
        """Full goal-market probabilities for each fixture row."""
        lambdas = self.predict_lambdas(X)
        out = []
        for h, a in zip(lambdas["home_lambda"], lambdas["away_lambda"]):
            out.append(match_probabilities(float(h), float(a)))
        return out


class CountWorker:
    """Generic count worker (corners, cards) using a regressor."""

    def __init__(self, target: str, backend: str = "auto", **kwargs: Any):
        self.target = target
        self.backend = backend
        self.model: Any = None
        self.feature_columns: List[str] = []

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> "CountWorker":
        self.feature_columns = list(X.columns)
        self.model = make_regressor(self.backend)
        self.model.fit(X, y[self.target].values)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X = X[self.feature_columns]
        return np.clip(self.model.predict(X), 0.0, None)

    def predict_probs(self, X: pd.DataFrame, line: float) -> np.ndarray:
        """P(over line) via Poisson around the predicted mean."""
        means = self.predict(X)
        from scipy.stats import poisson
        return 1.0 - poisson.cdf(int(line), means)


class BasketballWorker:
    """Predicts expected points for home & away teams in a basketball match.

    Uses the same leakage-safe feature matrix as the football workers but
    targets POINTS instead of goals. Probabilities (moneyline, spread,
    totals) are derived with the normal approximation in
    :mod:`src.models.basketball`.
    """

    def __init__(self, backend: str = "auto", **kwargs: Any):
        self.backend = backend
        self.model_home: Any = None
        self.model_away: Any = None
        self.feature_columns: List[str] = []

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> "BasketballWorker":
        self.feature_columns = list(X.columns)
        self.model_home = make_regressor(self.backend)
        self.model_away = make_regressor(self.backend)
        self.model_home.fit(X, y["home_goals"].values)
        self.model_away.fit(X, y["away_goals"].values)
        return self

    def predict_points(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Return predicted expected points for each row."""
        X = X[self.feature_columns]
        home_lambda = np.clip(self.model_home.predict(X), 60.0, 160.0)
        away_lambda = np.clip(self.model_away.predict(X), 60.0, 160.0)
        return {"home_lambda": home_lambda, "away_lambda": away_lambda}

    def predict_probs(self, X: pd.DataFrame, spread: float = 0.0,
                      total_line: float | None = None) -> List[Dict[str, float]]:
        """Full basketball market probabilities for each fixture row."""
        from src.models.basketball import basketball_probabilities

        pts = self.predict_points(X)
        out = []
        for h, a in zip(pts["home_lambda"], pts["away_lambda"]):
            out.append(basketball_probabilities(float(h), float(a),
                                                spread=spread, total_line=total_line))
        return out

