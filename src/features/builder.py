"""Feature builder — orchestrates all feature modules into a feature matrix.

CRITICAL: features for a match use ONLY data available before that match's
kickoff (no leakage). The builder computes features chronologically.
"""
from __future__ import annotations

from typing import Dict, List

import pandas as pd

from src.data.schemas import Fixture, MatchResult
from src.features.elo import EloSystem
from src.features.h2h import H2HFeatureBuilder
from src.features.rolling import RollingFeatureBuilder

# Canonical feature column order (stable ordering for training + inference).
FEATURE_COLUMNS = [
    "elo_diff", "home_elo", "away_elo",
    "home_xgf_ema_5", "away_xgf_ema_5", "home_xga_ema_5", "away_xga_ema_5",
    "home_scored_rate_5", "away_scored_rate_5",
    "home_ga_ema_5", "away_ga_ema_5",
    "h2h_home_win_ratio", "h2h_away_win_ratio", "h2h_home_goal_diff",
    "venue_home_undefeated_streak", "venue_home_scored_rate", "venue_home_conceded_rate",
    "home_kpai", "away_kpai", "xgf_std_10_home", "xgf_std_10_away",
]


class FeatureBuilder:
    """Builds leakage-safe feature matrices from history + fixtures."""

    def __init__(self, history: List[MatchResult], teams: Dict[str, object] | None = None,
                 players: Dict[str, object] | None = None):
        self.history = sorted(history, key=lambda m: m.kickoff)
        self.teams = teams or {}
        self.players = players or {}
        self.elo = EloSystem()
        self.rolling = RollingFeatureBuilder()
        self.h2h = H2HFeatureBuilder()
        self._fit_elo()

    def _fit_elo(self) -> None:
        """Fit Elo ratings over the full history (chronological)."""
        self.elo = EloSystem()
        self.elo.fit(self.history)

    # ------------------------------------------------------------------ #
    # Per-fixture feature vector
    # ------------------------------------------------------------------ #
    def fixture_features(self, fixture: Fixture) -> Dict[str, float]:
        """Build a feature dict for an upcoming fixture (no leakage)."""
        home_id, away_id = fixture.home_team_id, fixture.away_team_id
        # Only matches before kickoff are usable. Normalize timezones so we
        # never crash comparing offset-naive vs offset-aware datetimes.
        kickoff = fixture.kickoff
        if kickoff.tzinfo is not None:
            kickoff = kickoff.replace(tzinfo=None)
        prior = []
        for m in self.history:
            mk = m.kickoff
            if mk.tzinfo is not None:
                mk = mk.replace(tzinfo=None)
            if mk < kickoff:
                prior.append(m)

        home_roll = self.rolling.team_features(prior, home_id)
        away_roll = self.rolling.team_features(prior, away_id)
        h2h = self.h2h.features(prior, home_id, away_id)

        feats: Dict[str, float] = {
            "elo_diff": self.elo.rating_diff(home_id, away_id),
            "home_elo": self.elo.rating(home_id),
            "away_elo": self.elo.rating(away_id),
            "home_xgf_ema_5": home_roll.get("xgf_ema_5", 0.0),
            "away_xgf_ema_5": away_roll.get("xgf_ema_5", 0.0),
            "home_xga_ema_5": home_roll.get("xga_ema_5", 0.0),
            "away_xga_ema_5": away_roll.get("xga_ema_5", 0.0),
            "home_scored_rate_5": home_roll.get("scored_rate_5", 0.0),
            "away_scored_rate_5": away_roll.get("scored_rate_5", 0.0),
            "home_ga_ema_5": home_roll.get("ga_ema_5", 0.0),
            "away_ga_ema_5": away_roll.get("ga_ema_5", 0.0),
            "h2h_home_win_ratio": h2h.get("h2h_home_win_ratio", 0.0),
            "h2h_away_win_ratio": h2h.get("h2h_away_win_ratio", 0.0),
            "h2h_home_goal_diff": h2h.get("h2h_home_goal_diff", 0.0),
            "venue_home_undefeated_streak": h2h.get("venue_home_undefeated_streak", 0.0),
            "venue_home_scored_rate": h2h.get("venue_home_scored_rate", 0.0),
            "venue_home_conceded_rate": h2h.get("venue_home_conceded_rate", 0.0),
            "home_kpai": fixture.home_kpai,
            "away_kpai": fixture.away_kpai,
            "xgf_std_10_home": home_roll.get("xgf_std_10", 0.0),
            "xgf_std_10_away": away_roll.get("xgf_std_10", 0.0),
        }
        return feats

    # ------------------------------------------------------------------ #
    # Training matrix
    # ------------------------------------------------------------------ #
    def training_matrix(self, history: List[MatchResult] | None = None) -> pd.DataFrame:
        """Build a leakage-safe training matrix from historical matches.

        Each row = one historical match; features use only data available
        before that match's kickoff. Targets are derived from the match.
        """
        history = history or self.history
        rows: List[Dict[str, float]] = []
        targets: List[Dict[str, float]] = []

        for match in history:
            prior = [m for m in self.history if m.kickoff < match.kickoff]
            home_roll = self.rolling.team_features(prior, match.home_team_id)
            away_roll = self.rolling.team_features(prior, match.away_team_id)
            h2h = self.h2h.features(prior, match.home_team_id, match.away_team_id)

            feats = {
                "elo_diff": self.elo.rating_diff(match.home_team_id, match.away_team_id),
                "home_elo": self.elo.rating(match.home_team_id),
                "away_elo": self.elo.rating(match.away_team_id),
                "home_xgf_ema_5": home_roll.get("xgf_ema_5", 0.0),
                "away_xgf_ema_5": away_roll.get("xgf_ema_5", 0.0),
                "home_xga_ema_5": home_roll.get("xga_ema_5", 0.0),
                "away_xga_ema_5": away_roll.get("xga_ema_5", 0.0),
                "home_scored_rate_5": home_roll.get("scored_rate_5", 0.0),
                "away_scored_rate_5": away_roll.get("scored_rate_5", 0.0),
                "home_ga_ema_5": home_roll.get("ga_ema_5", 0.0),
                "away_ga_ema_5": away_roll.get("ga_ema_5", 0.0),
                "h2h_home_win_ratio": h2h.get("h2h_home_win_ratio", 0.0),
                "h2h_away_win_ratio": h2h.get("h2h_away_win_ratio", 0.0),
                "h2h_home_goal_diff": h2h.get("h2h_home_goal_diff", 0.0),
                "venue_home_undefeated_streak": h2h.get("venue_home_undefeated_streak", 0.0),
                "venue_home_scored_rate": h2h.get("venue_home_scored_rate", 0.0),
                "venue_home_conceded_rate": h2h.get("venue_home_conceded_rate", 0.0),
                "home_kpai": match.home_kpai,
                "away_kpai": match.away_kpai,
                "xgf_std_10_home": home_roll.get("xgf_std_10", 0.0),
                "xgf_std_10_away": away_roll.get("xgf_std_10", 0.0),
            }
            rows.append(feats)
            targets.append({
                "home_goals": float(match.home_goals),
                "away_goals": float(match.away_goals),
                "total_goals": float(match.total_goals),
                "outcome": match.outcome,
                "home_corners": float(match.home_corners),
                "away_corners": float(match.away_corners),
                "home_cards": float(match.home_cards),
                "away_cards": float(match.away_cards),
                "kickoff": match.kickoff,
                "match_id": match.match_id,
            })

        X = pd.DataFrame(rows, columns=FEATURE_COLUMNS)
        y = pd.DataFrame(targets)
        return X, y
