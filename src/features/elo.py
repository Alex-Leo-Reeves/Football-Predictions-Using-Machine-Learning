"""Elo rating system with margin-of-victory adjustment.

Tracks a dynamic strength rating per team, updated after every match
based on result and goal margin. This is a core "unpriced signal" that
captures true team strength better than raw form.
"""
from __future__ import annotations

from typing import Dict

from src.config import get
from src.data.schemas import MatchResult


class EloSystem:
    def __init__(self, k: float | None = None, home_advantage: float | None = None,
                 margin_factor: float | None = None, init_rating: float | None = None):
        self.k = k if k is not None else float(get("features.elo.k", 32))
        self.home_advantage = home_advantage if home_advantage is not None else float(
            get("features.elo.home_advantage", 65))
        self.margin_factor = margin_factor if margin_factor is not None else float(
            get("features.elo.margin_factor", 1.0))
        self.init_rating = init_rating if init_rating is not None else float(
            get("features.elo.init_rating", 1500))
        self.ratings: Dict[str, float] = {}

    def _rating(self, team_id: str) -> float:
        return self.ratings.get(team_id, self.init_rating)

    @staticmethod
    def _expected(rating_a: float, rating_b: float) -> float:
        return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))

    @staticmethod
    def _margin_multiplier(goal_diff: int) -> float:
        """Scale K by goal margin (cap at 3+)."""
        return min(3.0, abs(goal_diff)) ** 0.8

    def update(self, match: MatchResult) -> None:
        """Update both teams' ratings after a completed match."""
        home = self._rating(match.home_team_id)
        away = self._rating(match.away_team_id)

        # Home advantage shifts the expected outcome
        expected_home = self._expected(home + self.home_advantage, away)
        expected_away = 1.0 - expected_home

        goal_diff = match.home_goals - match.away_goals
        if goal_diff > 0:
            actual_home, actual_away = 1.0, 0.0
        elif goal_diff < 0:
            actual_home, actual_away = 0.0, 1.0
        else:
            actual_home, actual_away = 0.5, 0.5

        margin = self._margin_multiplier(goal_diff)
        k_home = self.k * margin * self.margin_factor
        k_away = self.k * margin * self.margin_factor

        self.ratings[match.home_team_id] = home + k_home * (actual_home - expected_home)
        self.ratings[match.away_team_id] = away + k_away * (actual_away - expected_away)

    def fit(self, history) -> None:
        """Fit ratings over chronological history."""
        for match in sorted(history, key=lambda m: m.kickoff):
            self.update(match)

    def rating(self, team_id: str) -> float:
        return self._rating(team_id)

    def rating_diff(self, home_id: str, away_id: str) -> float:
        """Home-adjusted rating difference (positive = home stronger)."""
        return (self._rating(home_id) + self.home_advantage) - self._rating(away_id)
