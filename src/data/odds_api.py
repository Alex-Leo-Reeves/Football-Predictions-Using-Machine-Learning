"""The Odds API client (api.the-odds-api.com).

Fallback odds source (OddsPapi is primary). Cache-aware with multi-key
rotation via KeyPool. Free tier = 500 requests/month per key.
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
DAILY_BUDGET = int(get("data.odds_api.daily_budget", 500))


class OddsAPIClient:
    def __init__(self, api_key: str | None = None, base_url: str | None = None,
                 daily_budget: int | None = None):
        from src.utils.key_rotation import KeyPool, parse_keys

        keys = parse_keys("ODDS_API_KEYS", "ODDS_API_KEY")
        if api_key and api_key not in keys:
            keys.insert(0, api_key)
        self.base_url = base_url or get("data.odds_api.base_url", "https://api.the-odds-api.com/v4")
        self.daily_budget = daily_budget or DAILY_BUDGET
        self.key_pool = KeyPool(
            keys,
            budget_per_key=self.daily_budget,
            state_file=CACHE_DIR / "odds_api_key_usage.json",
        )
        if not keys:
            raise RuntimeError("ODDS_API_KEY not set")
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

    @property
    def requests_left(self) -> int:
        return self.key_pool.requests_left

    @property
    def n_keys(self) -> int:
        return len(self.key_pool)

    def _get(self, path: str, params: Dict[str, Any], cache_key: str,
             ttl_hours: int = 6) -> Optional[Dict[str, Any]]:
        cache_file = CACHE_DIR / f"odds_{cache_key}.json"
        if cache_file.exists():
            try:
                cached = json.loads(cache_file.read_text())
                age_h = (datetime.now() - datetime.fromisoformat(cached.get("_fetched", "2000-01-01"))).total_seconds() / 3600
                if age_h <= ttl_hours:
                    return cached.get("data")
            except (json.JSONDecodeError, KeyError, ValueError):
                pass

        url = f"{self.base_url}/{path}"
        attempts = max(1, len(self.key_pool))
        for _ in range(attempts):
            key = self.key_pool.acquire()
            if key is None:
                print(f"[odds-api] all keys exhausted ({self.key_pool.requests_left} left)")
                return None
            try:
                resp = requests.get(url, params={**params, "apiKey": key}, timeout=20)
                resp.raise_for_status()
                data = resp.json()
                self.key_pool.record_use(key)
                cache_file.write_text(json.dumps({
                    "_fetched": datetime.now().isoformat(),
                    "data": data,
                }))
                return data
            except requests.RequestException as exc:
                print(f"[odds-api] request failed {path}: {exc}")
                return None
        return None

    def sports(self) -> List[Dict[str, Any]]:
        data = self._get("sports", {}, "sports", ttl_hours=24)
        return data or []

    def odds(self, sport: str, regions: str = "eu",
             markets: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Current odds for a sport across bookmakers."""
        markets = markets or get("data.odds_api.markets", ["h2h", "totals", "spreads"])
        params = {
            "regions": regions,
            "markets": ",".join(markets),
            "oddsFormat": "decimal",
        }
        data = self._get(f"sports/{sport}/odds", params, f"odds_{sport}_{regions}", ttl_hours=3)
        return data or []


def parse_odds_payload(items: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """Flatten The Odds API response into {match_key: {market_key: odds}}.

    ``match_key`` is ``"home_name @ away_name"`` (lowercased). For each
    match we keep the best (lowest) decimal odds across bookmakers per
    market, keyed by our internal market keys:
      h2h -> home_win / draw / away_win
      totals -> over_0_5 ... over_4_5 / under_...
      spreads -> home_spread / away_spread
    """
    out: Dict[str, Dict[str, float]] = {}
    for item in items:
        home = item.get("home_team", "")
        away = item.get("away_team", "")
        key = f"{home.lower()} @ {away.lower()}"
        match_odds: Dict[str, float] = {}
        for bm in item.get("bookmakers", []):
            for market in bm.get("markets", []):
                mkey = market.get("key")
                for outcome in market.get("outcomes", []):
                    name = outcome.get("name", "")
                    price = float(outcome.get("price", 0.0))
                    if price <= 1.0:
                        continue
                    if mkey == "h2h":
                        # NOTE: h2h outcomes are named after the TEAMS
                        # (e.g. "Arsenal", "Leeds United", "Draw"), not
                        # "Home"/"Away".
                        if name == home:
                            map_key = "home_win"
                        elif name == away:
                            map_key = "away_win"
                        elif name == "Draw":
                            map_key = "draw"
                        else:
                            map_key = None
                    elif mkey == "totals":
                        point = outcome.get("point")
                        if point is not None:
                            map_key = f"{'over' if name == 'Over' else 'under'}_{point:.1f}"
                        else:
                            map_key = None
                    else:
                        map_key = None
                    if map_key and (map_key not in match_odds or price < match_odds[map_key]):
                        match_odds[map_key] = price
        if match_odds:
            out[key] = match_odds
    return out
