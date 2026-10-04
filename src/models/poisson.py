"""Dixon-Coles bivariate Poisson goal model.

Given expected goals for home and away, builds the full scoreline
probability matrix (with the Dixon-Coles low-score correction) and
derives every goal-based market probability from it.
"""
from __future__ import annotations

import math
from typing import Dict

import numpy as np

from src.config import get


def _dc_tau(h: int, a: int, rho: float = 0.35) -> float:
    """Dixon-Coles tau correction for low scores."""
    if h == 0 and a == 0:
        return 1.0 - rho * rho
    if h == 0 and a == 1:
        return 1.0 + rho
    if h == 1 and a == 0:
        return 1.0 + rho
    if h == 1 and a == 1:
        return 1.0 - rho
    return 1.0


def score_matrix(lambda_home: float, lambda_away: float,
                 max_goals: int | None = None) -> np.ndarray:
    """Bivariate Poisson scoreline matrix (home x away)."""
    max_goals = max_goals or int(get("markets.max_goals", 6))
    probs = np.zeros((max_goals + 1, max_goals + 1))
    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            p_h = math.exp(-lambda_home) * lambda_home**h / math.factorial(h)
            p_a = math.exp(-lambda_away) * lambda_away**a / math.factorial(a)
            probs[h, a] = p_h * p_a * _dc_tau(h, a)
    total = probs.sum()
    if total > 0:
        probs /= total
    return probs


def match_probabilities(lambda_home: float, lambda_away: float,
                        max_goals: int | None = None) -> Dict[str, float]:
    """Derive all core market probabilities from expected goals."""
    mat = score_matrix(lambda_home, lambda_away, max_goals)
    n = mat.shape[0]

    # mat[h, a]: h = home goals (rows), a = away goals (cols)
    # home wins when h > a  -> strictly below the diagonal
    # away wins when a > h  -> strictly above the diagonal
    p_home = float(np.sum(np.tril(mat, -1)))
    p_draw = float(np.trace(mat))
    p_away = float(np.sum(np.triu(mat, 1)))

    # Goal totals
    total_probs = np.zeros(2 * n - 1)
    for h in range(n):
        for a in range(n):
            total_probs[h + a] += mat[h, a]

    def over(line: float) -> float:
        # P(total > line)
        idx = int(math.floor(line + 1))
        return float(total_probs[idx:].sum()) if idx < len(total_probs) else 0.0

    def team_over(team_goals: np.ndarray, line: float) -> float:
        idx = int(math.floor(line + 1))
        return float(team_goals[idx:].sum()) if idx < len(team_goals) else 0.0

    home_goals_dist = mat.sum(axis=1)
    away_goals_dist = mat.sum(axis=0)

    probs: Dict[str, float] = {
        "home_win": p_home,
        "draw": p_draw,
        "away_win": p_away,
        "home_or_draw": p_home + p_draw,
        "away_or_draw": p_away + p_draw,
        "home_or_away": p_home + p_away,
        "over_0_5": over(0.5),
        "over_1_5": over(1.5),
        "over_2_5": over(2.5),
        "over_3_5": over(3.5),
        "over_4_5": over(4.5),
        "under_0_5": 1.0 - over(0.5),
        "under_1_5": 1.0 - over(1.5),
        "under_2_5": 1.0 - over(2.5),
        "under_3_5": 1.0 - over(3.5),
        "under_4_5": 1.0 - over(4.5),
        "home_over_0_5": team_over(home_goals_dist, 0.5),
        "home_over_1_5": team_over(home_goals_dist, 1.5),
        "home_over_2_5": team_over(home_goals_dist, 2.5),
        "away_over_0_5": team_over(away_goals_dist, 0.5),
        "away_over_1_5": team_over(away_goals_dist, 1.5),
        "away_over_2_5": team_over(away_goals_dist, 2.5),
        "btts_yes": float(np.sum(mat[1:, 1:])),
        "btts_no": 1.0 - float(np.sum(mat[1:, 1:])),
        "home_clean_sheet": float(np.sum(mat[:, 0])),
        "away_clean_sheet": float(np.sum(mat[0, :])),
        "expected_total_goals": float(lambda_home + lambda_away),
    }
    return probs
