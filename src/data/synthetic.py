"""Synthetic historical match generator.

Generates a realistic (but synthetic) league history so the entire
pipeline can be trained and validated end-to-end without any API keys.

The generator models real football dynamics:
  * team strengths drawn from a distribution
  * home advantage
  * Poisson goal counts (with Dixon-Coles low-score adjustment)
  * xG with noise around goals
  * corners/cards correlated with team style
  * occasional key-player absences affecting team strength
"""
from __future__ import annotations

import math
import random
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import numpy as np

from src.data.schemas import Fixture, MatchResult, Player, Team


def _dc_tau(h: int, a: int) -> float:
    """Dixon-Coles tau correction for low scores (0-0, 1-0, 0-1, 1-1)."""
    if h == 0 and a == 0:
        return 1.0 - 0.35 * 0.35
    if h == 0 and a == 1:
        return 1.0 + 0.35
    if h == 1 and a == 0:
        return 1.0 + 0.35
    if h == 1 and a == 1:
        return 1.0 - 0.35
    return 1.0


def _sample_goals(lambda_home: float, lambda_away: float, rng: np.random.Generator) -> Tuple[int, int]:
    """Sample a scoreline from a bivariate Poisson with DC adjustment."""
    max_goals = 8
    probs = np.zeros((max_goals + 1, max_goals + 1))
    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            p_h = math.exp(-lambda_home) * lambda_home**h / math.factorial(h)
            p_a = math.exp(-lambda_away) * lambda_away**a / math.factorial(a)
            probs[h, a] = p_h * p_a * _dc_tau(h, a)
    probs /= probs.sum()
    flat = probs.ravel()
    idx = rng.choice(len(flat), p=flat)
    return int(idx // (max_goals + 1)), int(idx % (max_goals + 1))


class SyntheticLeague:
    """Generates a full multi-season league history + today's fixtures."""

    def __init__(self, n_teams: int = 20, n_seasons: int = 3, matches_per_season: int = 38,
                 seed: int = 42, today: datetime | None = None):
        self.rng = np.random.default_rng(seed)
        self.random = random.Random(seed)
        self.today = today or datetime(2026, 10, 4, 12, 0, 0)
        self.teams = self._make_teams(n_teams)
        self.players = self._make_players()
        self.history: List[MatchResult] = []
        self._build_history(n_seasons, matches_per_season)

    # ------------------------------------------------------------------ #
    # Team & player generation
    # ------------------------------------------------------------------ #
    def _make_teams(self, n: int) -> Dict[str, Team]:
        names = ["Alpha FC", "Beta United", "Gamma City", "Delta Rovers", "Epsilon Athletic",
                 "Zeta Wanderers", "Eta Town", "Theta Borough", "Iota County", "Kappa Rangers",
                 "Lambda Villa", "Mu Albion", "Nu Thistle", "Xi Harriers", "Omicron Vale",
                 "Pi Forest", "Rho Argyle", "Sigma Orient", "Tau Rovers", "Upsilon Dons"]
        teams: Dict[str, Team] = {}
        for i in range(n):
            name = names[i % len(names)]
            tid = f"T{i:02d}"
            strength = float(np.clip(self.rng.normal(1500, 120), 1250, 1750))
            teams[tid] = Team(id=tid, name=name, country="SYN", base_rating=strength)
        return teams

    def _make_players(self) -> Dict[str, Player]:
        players: Dict[str, Player] = {}
        pid = 0
        for tid, team in self.teams.items():
            for i in range(18):  # 18-man squad per team
                pid += 1
                players[f"P{pid:04d}"] = Player(
                    id=f"P{pid:04d}",
                    name=f"{team.name} Player {i+1}",
                    team_id=tid,
                    position=self.random.choice(["GK", "DF", "MF", "FW"]),
                    xg_per90=float(np.clip(self.rng.normal(0.15, 0.12), 0.0, 1.2)),
                    assists_per90=float(np.clip(self.rng.normal(0.1, 0.08), 0.0, 0.8)),
                    progressive_passes_per90=float(np.clip(self.rng.normal(2.0, 1.5), 0.0, 8.0)),
                    defensive_actions_per90=float(np.clip(self.rng.normal(3.0, 2.0), 0.0, 10.0)),
                    minutes_per90=float(self.rng.uniform(0.3, 1.0)),
                )
        return players

    # ------------------------------------------------------------------ #
    # History generation
    # ------------------------------------------------------------------ #
    def _team_attack(self, tid: str) -> float:
        """Attack strength derived from team base rating."""
        return (self.teams[tid].base_rating - 1500) / 150.0  # ~ -1.6 .. +1.6

    def _team_defence(self, tid: str) -> float:
        return -(self.teams[tid].base_rating - 1500) / 150.0

    def _build_history(self, n_seasons: int, matches_per_season: int) -> None:
        team_ids = list(self.teams.keys())
        match_id = 0
        season_start = self.today - timedelta(days=365 * n_seasons)

        for season in range(n_seasons):
            fixtures = []
            for i, home in enumerate(team_ids):
                for away in team_ids:
                    if home == away:
                        continue
                    fixtures.append((home, away))
            self.random.shuffle(fixtures)
            fixtures = fixtures[:matches_per_season]

            for home_id, away_id in fixtures:
                match_id += 1
                kickoff = season_start + timedelta(days=season * 365 + (match_id % 300))
                result = self._simulate_match(
                    match_id=f"M{match_id:05d}",
                    kickoff=kickoff,
                    home_id=home_id,
                    away_id=away_id,
                )
                self.history.append(result)

        self.history.sort(key=lambda m: m.kickoff)

    def _simulate_match(self, match_id: str, kickoff: datetime,
                        home_id: str, away_id: str) -> MatchResult:
        home_attack = self._team_attack(home_id)
        away_attack = self._team_attack(away_id)
        home_defence = self._team_defence(home_id)
        away_defence = self._team_defence(away_id)

        home_kpai = float(self.rng.uniform(0.7, 1.0))
        away_kpai = float(self.rng.uniform(0.7, 1.0))

        lambda_home = max(0.1, 1.35 + home_attack + away_defence + 0.25 * (home_kpai - 0.85))
        lambda_away = max(0.1, 1.05 + away_attack + home_defence + 0.25 * (away_kpai - 0.85))

        home_goals, away_goals = _sample_goals(lambda_home, lambda_away, self.rng)

        home_xg = max(0.0, float(self.rng.normal(lambda_home, 0.35)))
        away_xg = max(0.0, float(self.rng.normal(lambda_away, 0.35)))

        home_corners = max(0, int(self.rng.normal(4.5 + home_attack * 2.0, 2.0)))
        away_corners = max(0, int(self.rng.normal(4.0 + away_attack * 2.0, 2.0)))

        home_cards = max(0, int(self.rng.normal(1.6, 0.8)))
        away_cards = max(0, int(self.rng.normal(1.6, 0.8)))

        return MatchResult(
            match_id=match_id,
            kickoff=kickoff,
            home_team_id=home_id,
            away_team_id=away_id,
            home_goals=home_goals,
            away_goals=away_goals,
            home_xg=home_xg,
            away_xg=away_xg,
            home_corners=home_corners,
            away_corners=away_corners,
            home_cards=home_cards,
            away_cards=away_cards,
            home_npxg_state_adj=home_xg,
            away_npxg_state_adj=away_xg,
            home_kpai=home_kpai,
            away_kpai=away_kpai,
        )

    # ------------------------------------------------------------------ #
    # Today's fixtures
    # ------------------------------------------------------------------ #
    def today_fixtures(self, n: int = 10) -> List[Fixture]:
        """Generate today's fixtures with plausible odds snapshots."""
        team_ids = list(self.teams.keys())
        fixtures: List[Fixture] = []
        used = set()
        for i in range(n):
            home_id, away_id = self.random.sample(team_ids, 2)
            if (home_id, away_id) in used:
                continue
            used.add((home_id, away_id))
            kickoff = self.today + timedelta(hours=2 + i)
            fixtures.append(Fixture(
                fixture_id=f"F{i:04d}",
                kickoff=kickoff,
                home_team_id=home_id,
                away_team_id=away_id,
                home_team_name=self.teams[home_id].name,
                away_team_name=self.teams[away_id].name,
                league="SYN League",
                country="SYN",
                home_kpai=float(self.rng.uniform(0.8, 1.0)),
                away_kpai=float(self.rng.uniform(0.8, 1.0)),
            ))
        return fixtures

    def team(self, tid: str) -> Team:
        return self.teams[tid]

    def players_for(self, tid: str) -> List[Player]:
        return [p for p in self.players.values() if p.team_id == tid]

