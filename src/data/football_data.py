"""football-data.org client (api.football-data.org).

Free tier: 12 competitions, 10 calls/minute — fixtures, results, standings.
The free tier has NO xG / corners / cards (those are paid add-ons), so match
statistics default to 0.0. Cache-aware with multi-key rotation via KeyPool
(``FOOTBALL_DATA_ORG_KEYS`` comma-separated or ``FOOTBALL_DATA_ORG_KEY``).
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from src.config import get

CACHE_DIR = Path(get("data.cache_dir", "data/raw"))
DAILY_BUDGET = int(get("data.football_data.daily_budget", 100))


class FootballDataClient:
    """football-data.org client with multi-key rotation + disk cache."""

    def __init__(self, api_key: str | None = None, base_url: str | None = None,
                 daily_budget: int | None = None):
        from src.utils.key_rotation import KeyPool, parse_keys

        keys = parse_keys("FOOTBALL_DATA_ORG_KEYS", "FOOTBALL_DATA_ORG_KEY")
        if api_key and api_key not in keys:
            keys.insert(0, api_key)
        self.base_url = base_url or get("data.football_data.base_url",
                                        "https://api.football-data.org/v4")
        self.daily_budget = daily_budget or DAILY_BUDGET
        self.key_pool = KeyPool(
            keys,
            budget_per_key=self.daily_budget,
            state_file=CACHE_DIR / "football_data_key_usage.json",
        )
        if not keys:
            raise RuntimeError("FOOTBALL_DATA_ORG_KEY not set")
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Cache-aware GET with key rotation
    # ------------------------------------------------------------------ #
    def _get(self, path: str, params: Dict[str, Any], cache_key: str,
             ttl_days: int = 7) -> Optional[Dict[str, Any]]:
        """GET with disk cache + key rotation. Returns parsed JSON or None."""
        cache_file = CACHE_DIR / f"fdata_{cache_key.replace('/', '_').replace('?', '_')}.json"
        if cache_file.exists():
            try:
                cached = json.loads(cache_file.read_text())
                age_days = (datetime.now() - datetime.fromisoformat(
                    cached.get("_fetched", "2000-01-01"))).days
                if age_days <= ttl_days:
                    return cached.get("data")
            except (json.JSONDecodeError, KeyError, ValueError):
                pass

        url = f"{self.base_url}/{path}"
        attempts = max(1, len(self.key_pool))
        for _ in range(attempts):
            key = self.key_pool.acquire()
            if key is None:
                print(f"[football-data] all keys exhausted ({self.key_pool.requests_left} left)")
                return None
            try:
                resp = requests.get(url, headers={"X-Auth-Token": key},
                                    params=params, timeout=20)
                resp.raise_for_status()
                data = resp.json()
                self.key_pool.record_use(key)
                cache_file.write_text(json.dumps({
                    "_fetched": datetime.now().isoformat(),
                    "data": data,
                }))
                return data
            except requests.RequestException as exc:
                print(f"[football-data] request failed {path}: {exc}")
                return None
        return None

    # ------------------------------------------------------------------ #
    # Endpoints
    # ------------------------------------------------------------------ #
    def competitions(self) -> List[Dict[str, Any]]:
        """All competitions visible to the key (free tier = 12)."""
        data = self._get("competitions", {}, "competitions", ttl_days=30)
        return (data or {}).get("competitions", [])

    def matches_by_competition(self, code: str, season: int) -> List[Dict[str, Any]]:
        """All matches for a competition code + season (includes final scores)."""
        data = self._get(f"competitions/{code}/matches", {"season": season},
                         f"comp_{code}_{season}", ttl_days=7)
        return (data or {}).get("matches", [])

    def matches_by_date(self, date_from: str, date_to: str) -> List[Dict[str, Any]]:
        """All matches across the free competitions between two dates."""
        data = self._get("matches", {"dateFrom": date_from, "dateTo": date_to},
                         f"matches_{date_from}_{date_to}", ttl_days=1)
        return (data or {}).get("matches", [])


# ---------------------------------------------------------------------- #
# Mapping helpers: football-data.org JSON -> our schemas
# ---------------------------------------------------------------------- #
def parse_football_data_match(item: Dict[str, Any]) -> Dict[str, Any]:
    """Extract the core fields from a football-data.org match JSON item."""
    comp = item.get("competition", {})
    area = comp.get("area", {})
    score = item.get("score", {}) or {}
    ft = score.get("fullTime", {}) or {}
    home = item.get("homeTeam", {}) or {}
    away = item.get("awayTeam", {}) or {}
    return {
        "fixture_id": item.get("id"),
        "kickoff": item.get("utcDate"),
        "status": item.get("status"),
        "home_id": home.get("id"),
        "away_id": away.get("id"),
        "home_name": home.get("name"),
        "away_name": away.get("name"),
        "home_goals": ft.get("home"),
        "away_goals": ft.get("away"),
        "league": comp.get("name"),
        "country": area.get("name"),
    }
