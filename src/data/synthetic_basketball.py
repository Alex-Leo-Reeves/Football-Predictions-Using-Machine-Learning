"""Synthetic basketball match generator.

Generates a realistic (but synthetic) basketball league history so the
pipeline can train a basketball worker and produce basketball selections
without any API keys.

Basketball differs from football:
  * high-scoring (teams score ~100-120 points)
  * no draws -> moneyline is home_win / away_win only
  * markets are moneyline, point spread, and over/under totals
  * points are modelled with a normal (Gaussian) approximation rather than
    Poisson, because scoring is continuous-ish with high variance.
"""
from __future__ import annotations

import random
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import numpy as np

from src.data.schemas import Fixture, MatchResult, Team


class SyntheticBasketballLeague:
    """Generates a multi-season basketball history + today's fixtures."""

    def __init__(self, n_teams: int = 16, n_seasons: int = 3, matches_per_season: int = 30,
                 seed: int = 7, today: datetime | None = None):
        self.rng = np.random.default_rng(seed)
        self.random = random.Random(seed)
        self.today = today or datetime(2026, 10, 4, 12, 0, 0)
        self.teams: Dict[str, Team] = self._make_teams(n_teams)
        self.history: List[MatchResult] = []
        self._build_history(n_seasons, matches_per_season)

    # ------------------------------------------------------------------ #
    # Team generation
    # ------------------------------------------------------------------ #
    def _make_teams(self, n: int) -> Dict[str, Team]:
        names = ["Hoop City", "Slam United", "Dunkville", "Brick County", "Alley-Oop FC",
                 "Swish Town", "Rebound Rovers", "Fastbreak City", "Post Up", "Zone Defense",
                 "Full Court", "Pick & Roll", "Iso Ball", "Bench Mob", "Sixth Man", "Glass Cleaners"]
        teams: Dict[str, Team] = {}
        for i in range(n):
            name = names[i % len(names)]
            tid = f"B{i:02d}"
            strength = float(np.clip(self.rng.normal(1500, 100), 1300, 1700))
            teams[tid] = Team(id=tid, name=name, country="SYN-BB", base_rating=strength)
        return teams

    # ------------------------------------------------------------------ #
    # History
    # ------------------------------------------------------------------ #
    def _build_history(self, n_seasons: int, matches_per_season: int) -> None:
        team_ids = list(self.teams.keys())
        season_start = datetime(self.today.year - n_seasons, 10, 1, 19, 0, 0)
        match_id = 0
        for season in range(n_seasons):
            for _ in range(matches_per_season):
                home_id, away_id = self.random.sample(team_ids, 2)
                kickoff = season_start + timedelta(days=season * 365 + (match_id % 200))
                result = self._simulate_match(
                    match_id=f"BM{match_id:05d}",
                    kickoff=kickoff,
                    home_id=home_id,
                    away_id=away_id,
                )
                self.history.append(result)
                match_id += 1
        self.history.sort(key=lambda m: m.kickoff)

    def _simulate_match(self, match_id: str, kickoff: datetime,
                        home_id: str, away_id: str) -> MatchResult:
        home_attack = self._team_attack(home_id)
        away_attack = self._team_attack(away_id)
        home_defence = self._team_defence(home_id)
        away_defence = self._team_defence(away_id)

        # Expected points per team (~105-120 typical NBA total).
        lambda_home = max(80.0, 108.0 + home_attack + away_defence)
        lambda_away = max(80.0, 104.0 + away_attack + home_defence)

        # Gaussian scoring with ~11-12 point standard deviation per team.
        home_pts = max(0, int(round(self.rng.normal(lambda_home, 11.0))))
        away_pts = max(0, int(round(self.rng.normal(lambda_away, 11.0))))

        return MatchResult(
            match_id=match_id,
            kickoff=kickoff,
            home_team_id=home_id,
            away_team_id=away_id,
            home_goals=home_pts,
            away_goals=away_pts,
            sport="basketball",
            home_xg=lambda_home,
            away_xg=lambda_away,
        )

    # ------------------------------------------------------------------ #
    # Strength helpers (mirror football synthetic)
    # ------------------------------------------------------------------ #
    def _team_attack(self, tid: str) -> float:
        return (self.teams[tid].base_rating - 1500.0) / 400.0

    def _team_defence(self, tid: str) -> float:
        return -(self.teams[tid].base_rating - 1500.0) / 400.0

    # ------------------------------------------------------------------ #
    # Today's fixtures
    # ------------------------------------------------------------------ #
    def today_fixtures(self, n: int = 10) -> List[Fixture]:
        """Generate today's basketball fixtures with plausible odds snapshots."""
        team_ids = list(self.teams.keys())
        fixtures: List[Fixture] = []
        used = set()
        attempts = 0
        while len(fixtures) < n and attempts < n * 10:
            attempts += 1
            home_id, away_id = self.random.sample(team_ids, 2)
            if (home_id, away_id) in used:
                continue
            used.add((home_id, away_id))
            kickoff = self.today + timedelta(hours=2 + len(fixtures))
            fixtures.append(Fixture(
                fixture_id=f"BF{len(fixtures):04d}",
                kickoff=kickoff,
                home_team_id=home_id,
                away_team_id=away_id,
                home_team_name=self.teams[home_id].name,
                away_team_name=self.teams[away_id].name,
                league="SYN Basketball",
                country="SYN-BB",
                sport="basketball",
            ))
        return fixtures

    def team(self, tid: str) -> Team:
        return self.teams[tid]
