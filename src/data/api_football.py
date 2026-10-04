"""API-Football client (v3.football.api-sports.io).

Cache-aware and budget-aware:
  * every response is cached to data/raw/ so we never re-fetch the same data
  * a daily request counter guards the free-tier 100 req/day limit
  * per-match statistics (xG, corners, cards) cost 1 request each, so the
    caller controls how many fixtures to enrich per run
"""
from __future__ import annotations

import json
import os
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from src.config import get

CACHE_DIR = Path(get("data.cache_dir", "data/raw"))
DAILY_BUDGET = int(get("data.football_api.daily_budget", 90))


class APIFootballClient:
    """API-Football client with multi-key rotation + disk cache.

    Keys come from ``FOOTBALL_API_KEYS`` (comma-separated) or
    ``FOOTBALL_API_KEY``. Each key gets its own daily budget; when one is
    exhausted the pool rotates to the next.
    """

    def __init__(self, api_key: str | None = None, base_url: str | None = None,
                 daily_budget: int | None = None):
        from src.utils.key_rotation import KeyPool, parse_keys

        keys = parse_keys("FOOTBALL_API_KEYS", "FOOTBALL_API_KEY")
        if api_key and api_key not in keys:
            keys.insert(0, api_key)
        self.base_url = base_url or get("data.football_api.base_url", "https://v3.football.api-sports.io")
        self.daily_budget = daily_budget or DAILY_BUDGET
        self.key_pool = KeyPool(
            keys,
            budget_per_key=self.daily_budget,
            state_file=CACHE_DIR / "football_key_usage.json",
        )
        if not keys:
            raise RuntimeError("FOOTBALL_API_KEY not set")
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

    @property
    def requests_left(self) -> int:
        return self.key_pool.requests_left

    @property
    def n_keys(self) -> int:
        return len(self.key_pool)

    # ------------------------------------------------------------------ #
    # Cache-aware GET with key rotation
    # ------------------------------------------------------------------ #
    def _get(self, path: str, params: Dict[str, Any], cache_key: str | None = None,
             ttl_days: int = 30) -> Optional[Dict[str, Any]]:
        """GET with disk cache + key rotation. Returns parsed JSON or None."""
        cache_key = cache_key or f"{path}_{json.dumps(params, sort_keys=True)}"
        cache_file = CACHE_DIR / f"{cache_key.replace('/', '_').replace('?', '_')}.json"

        # Serve from cache if fresh
        if cache_file.exists():
            try:
                cached = json.loads(cache_file.read_text())
                age_days = (datetime.now() - datetime.fromisoformat(cached.get("_fetched", "2000-01-01"))).days
                if age_days <= ttl_days:
                    return cached.get("data")
            except (json.JSONDecodeError, KeyError, ValueError):
                pass

        url = f"{self.base_url}/{path}"
        attempts = max(1, len(self.key_pool))
        for _ in range(attempts):
            key = self.key_pool.acquire()
            if key is None:
                print(f"[api-football] all keys exhausted ({self.key_pool.requests_left} requests left)")
                return None
            try:
                resp = requests.get(url, headers={"x-apisports-key": key},
                                    params=params, timeout=20)
                resp.raise_for_status()
                data = resp.json()
                self.key_pool.record_use(key)
                if data.get("errors"):
                    # rate-limit / quota error -> try next key
                    print(f"[api-football] API error for {path}: {data['errors']}")
                    continue
                cache_file.write_text(json.dumps({
                    "_fetched": datetime.now().isoformat(),
                    "data": data,
                }))
                return data
            except requests.RequestException as exc:
                print(f"[api-football] request failed {path}: {exc}")
                return None
        return None

    # ------------------------------------------------------------------ #
    # Endpoints
    # ------------------------------------------------------------------ #
    def fixtures_by_season(self, league: int, season: int) -> List[Dict[str, Any]]:
        """All fixtures for a league+season (includes final scores)."""
        data = self._get("fixtures", {"league": league, "season": season},
                         cache_key=f"fixtures_{league}_{season}", ttl_days=7)
        return (data or {}).get("response", [])

    def fixture_statistics(self, fixture_id: int) -> List[Dict[str, Any]]:
        """Per-match statistics (xG, corners, cards) for one fixture."""
        data = self._get("fixtures/statistics", {"fixture": fixture_id},
                         cache_key=f"stats_{fixture_id}", ttl_days=365)
        return (data or {}).get("response", [])

    def today_fixtures(self, league: int, season: int) -> List[Dict[str, Any]]:
        """Fixtures scheduled today for a league."""
        today = date.today().isoformat()
        data = self._get("fixtures", {"league": league, "season": season, "date": today},
                         cache_key=f"fixtures_today_{league}_{today}", ttl_days=1)
        return (data or {}).get("response", [])

    def lineups(self, fixture_id: int) -> List[Dict[str, Any]]:
        data = self._get("fixtures/lineups", {"fixture": fixture_id},
                         cache_key=f"lineups_{fixture_id}", ttl_days=1)
        return (data or {}).get("response", [])

    def injuries(self, fixture_id: int) -> List[Dict[str, Any]]:
        data = self._get("injuries", {"fixture": fixture_id},
                         cache_key=f"injuries_{fixture_id}", ttl_days=1)
        return (data or {}).get("response", [])

    def h2h(self, team1: int, team2: int) -> List[Dict[str, Any]]:
        data = self._get("fixtures/headtohead", {"h2h": f"{team1}-{team2}"},
                         cache_key=f"h2h_{team1}_{team2}", ttl_days=7)
        return (data or {}).get("response", [])

    def standings(self, league: int, season: int) -> List[Dict[str, Any]]:
        data = self._get("standings", {"league": league, "season": season},
                         cache_key=f"standings_{league}_{season}", ttl_days=1)
        return (data or {}).get("response", [])


# ---------------------------------------------------------------------- #
# Mapping helpers: API-Football JSON -> our schemas
# ---------------------------------------------------------------------- #
def parse_fixture_basic(item: Dict[str, Any]) -> Dict[str, Any]:
    """Extract the core fields from a fixture JSON item."""
    fixture = item.get("fixture", {})
    teams = item.get("teams", {})
    goals = item.get("goals", {})
    league = item.get("league", {})
    return {
        "fixture_id": fixture.get("id"),
        "kickoff": fixture.get("date"),
        "status": fixture.get("status", {}).get("short"),
        "home_id": teams.get("home", {}).get("id"),
        "away_id": teams.get("away", {}).get("id"),
        "home_name": teams.get("home", {}).get("name"),
        "away_name": teams.get("away", {}).get("name"),
        "home_goals": goals.get("home"),
        "away_goals": goals.get("away"),
        "league_id": league.get("id"),
        "league_name": league.get("name"),
        "season": league.get("season"),
    }


def parse_statistics(stats: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Extract xG / corners / cards from a fixture-statistics response.

    ``stats`` is the list of per-team stat blocks:
    [{"team": {...}, "statistics": [{"type": "...", "value": ...}, ...]}, ...]
    """
    out: Dict[str, Any] = {"home": {}, "away": {}}
    if len(stats) >= 2:
        for side, block in zip(("home", "away"), stats[:2]):
            for stat in block.get("statistics", []):
                stype = stat.get("type", "")
                value = stat.get("value")
                if stype in ("expected_goals", "Corner Kicks", "Yellow Cards",
                             "Red Cards", "Total Shots", "Shots on Goal", "Fouls"):
                    out[side][stype] = value
    return out

