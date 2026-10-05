"""Head-to-head (H2H) and venue features.

Captures matchup-specific history: win rate vs this opponent, goal
differential vs this opponent, and the "venue fortress" index (home
undefeated streak, home scoring consistency).
"""
from __future__ import annotations

from typing import Dict, List

from src.config import get
from src.data.schemas import MatchResult, naive


class H2HFeatureBuilder:
    def __init__(self, window: int | None = None):
        self.window = window or int(get("features.h2h.window", 10))

    def features(self, history: List[MatchResult], home_id: str, away_id: str) -> Dict[str, float]:
        """H2H + venue features for a home/away pairing."""
        meetings = [
            m for m in history
            if {m.home_team_id, m.away_team_id} == {home_id, away_id}
        ]
        meetings = sorted(meetings, key=lambda m: naive(m.kickoff))[-self.window:]

        h2h_home_wins = 0
        h2h_away_wins = 0
        h2h_draws = 0
        h2h_home_goals = 0
        h2h_away_goals = 0
        for m in meetings:
            if m.home_team_id == home_id:
                h2h_home_goals += m.home_goals
                h2h_away_goals += m.away_goals
                if m.home_goals > m.away_goals:
                    h2h_home_wins += 1
                elif m.home_goals < m.away_goals:
                    h2h_away_wins += 1
                else:
                    h2h_draws += 1
            else:
                h2h_home_goals += m.away_goals
                h2h_away_goals += m.home_goals
                if m.away_goals > m.home_goals:
                    h2h_home_wins += 1
                elif m.away_goals < m.home_goals:
                    h2h_away_wins += 1
                else:
                    h2h_draws += 1

        n = len(meetings)
        feats: Dict[str, float] = {
            "h2h_n": float(n),
            "h2h_home_win_ratio": (h2h_home_wins / n) if n else 0.0,
            "h2h_away_win_ratio": (h2h_away_wins / n) if n else 0.0,
            "h2h_draw_ratio": (h2h_draws / n) if n else 0.0,
            "h2h_home_goals_avg": (h2h_home_goals / n) if n else 0.0,
            "h2h_away_goals_avg": (h2h_away_goals / n) if n else 0.0,
            "h2h_home_goal_diff": (h2h_home_goals - h2h_away_goals) / n if n else 0.0,
        }

        # ---- Venue fortress index (home team at home) ----
        home_matches = sorted(
            [m for m in history if m.home_team_id == home_id],
            key=lambda m: naive(m.kickoff),
        )
        undefeated_streak = 0
        for m in reversed(home_matches):
            if m.home_goals >= m.away_goals:
                undefeated_streak += 1
            else:
                break
        home_scored = [1.0 if m.home_goals > 0 else 0.0 for m in home_matches[-20:]]
        home_conceded = [1.0 if m.away_goals > 0 else 0.0 for m in home_matches[-20:]]

        feats["venue_home_undefeated_streak"] = float(undefeated_streak)
        feats["venue_home_scored_rate"] = float(sum(home_scored) / len(home_scored)) if home_scored else 0.0
        feats["venue_home_conceded_rate"] = float(sum(home_conceded) / len(home_conceded)) if home_conceded else 0.0
        feats["venue_home_ppg"] = self._ppg(home_matches[-20:])
        return feats

    @staticmethod
    def _ppg(matches: List[MatchResult]) -> float:
        if not matches:
            return 0.0
        pts = 0
        for m in matches:
            if m.home_goals > m.away_goals:
                pts += 3
            elif m.home_goals == m.away_goals:
                pts += 1
        return pts / len(matches)
