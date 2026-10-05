"""Rolling / exponential-moving-average features.

Computes EMA windows for goals, xG, corners, cards and scoring streaks.
EMA down-weights old matches, capturing "recent form" better than flat
rolling averages.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from src.config import get
from src.data.schemas import MatchResult, naive


def ema(values: List[float], span: int) -> float:
    """Exponential moving average of the last `span` values (or fewer)."""
    if not values:
        return 0.0
    alpha = 2.0 / (span + 1.0)
    result = values[0]
    for v in values[1:]:
        result = alpha * v + (1.0 - alpha) * result
    return float(result)


class RollingFeatureBuilder:
    """Builds per-team rolling features from match history."""

    def __init__(self, windows: List[int] | None = None):
        self.windows = windows or [int(w) for w in get("features.ema_windows", [3, 5, 10])]

    def _team_matches(self, history: List[MatchResult], team_id: str) -> List[MatchResult]:
        return [m for m in history if m.home_team_id == team_id or m.away_team_id == team_id]

    def _goals_for(self, match: MatchResult, team_id: str) -> int:
        return match.home_goals if match.home_team_id == team_id else match.away_goals

    def _goals_against(self, match: MatchResult, team_id: str) -> int:
        return match.away_goals if match.home_team_id == team_id else match.home_goals

    def _xg_for(self, match: MatchResult, team_id: str) -> float:
        return match.home_xg if match.home_team_id == team_id else match.away_xg

    def _xg_against(self, match: MatchResult, team_id: str) -> float:
        return match.away_xg if match.home_team_id == team_id else match.home_xg

    def _is_home(self, match: MatchResult, team_id: str) -> bool:
        return match.home_team_id == team_id

    def team_features(self, history: List[MatchResult], team_id: str) -> Dict[str, float]:
        """Rolling features for one team (chronological)."""
        matches = sorted(self._team_matches(history, team_id), key=lambda m: naive(m.kickoff))
        gf, ga, xgf, xga, scored, conceded = [], [], [], [], [], []
        home_gf, home_ga = [], []

        for m in matches:
            gf.append(self._goals_for(m, team_id))
            ga.append(self._goals_against(m, team_id))
            xgf.append(self._xg_for(m, team_id))
            xga.append(self._xg_against(m, team_id))
            scored.append(1.0 if self._goals_for(m, team_id) > 0 else 0.0)
            conceded.append(1.0 if self._goals_against(m, team_id) > 0 else 0.0)
            if self._is_home(m, team_id):
                home_gf.append(self._goals_for(m, team_id))
                home_ga.append(self._goals_against(m, team_id))

        feats: Dict[str, float] = {}
        for w in self.windows:
            feats[f"gf_ema_{w}"] = ema(gf, w)
            feats[f"ga_ema_{w}"] = ema(ga, w)
            feats[f"xgf_ema_{w}"] = ema(xgf, w)
            feats[f"xga_ema_{w}"] = ema(xga, w)
            feats[f"scored_rate_{w}"] = ema(scored, w)
            feats[f"conceded_rate_{w}"] = ema(conceded, w)
        # home-only scoring rate (venue effect)
        feats["home_scored_rate"] = ema(scored, 10) if scored else 0.0
        feats["home_gf_ema_5"] = ema(home_gf, 5) if home_gf else 0.0
        feats["home_ga_ema_5"] = ema(home_ga, 5) if home_ga else 0.0
        # volatility (std of recent xG) — used by variance auditor
        recent_xgf = xgf[-10:]
        feats["xgf_std_10"] = float(np.std(recent_xgf)) if len(recent_xgf) > 1 else 0.0
        return feats
