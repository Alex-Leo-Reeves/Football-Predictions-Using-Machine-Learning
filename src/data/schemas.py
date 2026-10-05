"""Lightweight data schemas (dataclasses) used across the pipeline.

Keeping schemas as plain dataclasses (rather than a heavy ORM) keeps the
local footprint tiny while remaining fully compatible with GitHub Actions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional


def naive(dt: datetime) -> datetime:
    """Return a timezone-naive copy of ``dt``.

    Strips tzinfo so naive/aware comparisons never raise
    ``TypeError: can't compare offset-naive and offset-aware datetimes``.
    """
    if dt.tzinfo is not None:
        return dt.replace(tzinfo=None)
    return dt


@dataclass
class Team:
    """A team with a stable identifier used across features."""
    id: str
    name: str
    country: str = ""
    base_rating: float = 1500.0


@dataclass
class Player:
    """A player with per-90 performance profile used for KPAI."""
    id: str
    name: str
    team_id: str
    position: str = ""
    # per-90 metrics (used to weight "importance")
    xg_per90: float = 0.0
    assists_per90: float = 0.0
    progressive_passes_per90: float = 0.0
    defensive_actions_per90: float = 0.0
    minutes_per90: float = 0.0


@dataclass
class MatchResult:
    """A completed match used for training features."""
    match_id: str
    kickoff: datetime
    home_team_id: str
    away_team_id: str
    home_goals: int
    away_goals: int
    # sport discriminator: "football" (goals) or "basketball" (points).
    # For basketball, home_goals/away_goals hold the final POINTS scored.
    sport: str = "football"
    home_xg: float = 0.0
    away_xg: float = 0.0
    home_corners: int = 0
    away_corners: int = 0
    home_cards: int = 0
    away_cards: int = 0
    # game-state normalized attacking metrics (see features/game_state.py)
    home_npxg_state_adj: float = 0.0
    away_npxg_state_adj: float = 0.0
    # lineup availability (0..1 fraction of key players available)
    home_kpai: float = 1.0
    away_kpai: float = 1.0
    # optional: which players started (ids)
    home_starters: List[str] = field(default_factory=list)
    away_starters: List[str] = field(default_factory=list)

    @property
    def total_goals(self) -> int:
        return self.home_goals + self.away_goals

    @property
    def outcome(self) -> str:
        """1X2 label."""
        if self.home_goals > self.away_goals:
            return "H"
        if self.home_goals == self.away_goals:
            return "D"
        return "A"


@dataclass
class Fixture:
    """An upcoming match to predict."""
    fixture_id: str
    kickoff: datetime
    home_team_id: str
    away_team_id: str
    home_team_name: str = ""
    away_team_name: str = ""
    league: str = ""
    country: str = ""
    # sport discriminator: "football" or "basketball"
    sport: str = "football"
    # expected lineup availability (0..1) — from probabilistic lineup model
    home_kpai: float = 1.0
    away_kpai: float = 1.0
    # risk flags set by hard veto rules
    risk_flags: List[str] = field(default_factory=list)
    # optional market odds snapshot {market_name: decimal_odds}
    odds: Dict[str, float] = field(default_factory=dict)

    @property
    def label(self) -> str:
        return f"{self.home_team_name or self.home_team_id} vs {self.away_team_name or self.away_team_id}"
