"""OddsPapi client (api.oddspapi.io).

Primary odds source — better than The Odds API for our use case because it
provides FREE historical odds with line-movement analysis (needed to train
the edge model), 300+ bookmakers, and a WebSocket feed.

Auth: ``apiKey`` query parameter. Multi-key rotation via KeyPool.
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
DAILY_BUDGET = int(get("data.oddspapi.daily_budget", 500))

# Fallback market-id -> name mapping (enriched by GET /v4/markets when
# available). 101 = moneyline, 104 = totals, 105 = spread, etc.
FALLBACK_MARKET_NAMES = {
    "101": "moneyline",
    "102": "double_chance",
    "103": "draw_no_bet",
    "104": "totals",
    "105": "spread",
    "106": "asian_handicap",
    "107": "team_totals",
    "108": "correct_score",
    "109": "both_to_score",
}


class OddsPapiClient:
    def __init__(self, api_key: str | None = None, base_url: str | None = None,
                 daily_budget: int | None = None):
        from src.utils.key_rotation import KeyPool, parse_keys

        keys = parse_keys("ODDSPAPI_KEYS", "ODDSPAPI_KEY")
        if api_key and api_key not in keys:
            keys.insert(0, api_key)
        self.base_url = base_url or get("data.oddspapi.base_url", "https://api.oddspapi.io")
        self.daily_budget = daily_budget or DAILY_BUDGET
        self.key_pool = KeyPool(
            keys,
            budget_per_key=self.daily_budget,
            state_file=CACHE_DIR / "oddspapi_key_usage.json",
        )
        if not keys:
            raise RuntimeError("ODDSPAPI_KEY not set")
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self._market_names: Dict[str, str] = dict(FALLBACK_MARKET_NAMES)
        # marketId -> {"marketType": str, "outcomes": {outcomeId: outcomeName},
        #              "handicap": float}
        self._market_defs: Dict[str, Dict[str, Any]] = {}

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
             ttl_hours: int = 6) -> Optional[Dict[str, Any]]:
        cache_key = cache_key or f"{path}_{json.dumps(params, sort_keys=True)}"
        # Hash long cache keys so the filename never exceeds filesystem limits.
        if len(cache_key) > 80:
            import hashlib
            cache_key = hashlib.sha256(cache_key.encode()).hexdigest()[:40]
        cache_file = CACHE_DIR / f"oddspapi_{cache_key.replace('/', '_').replace('?', '_')}.json"

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
                print(f"[oddspapi] all keys exhausted ({self.key_pool.requests_left} left)")
                return None
            try:
                resp = requests.get(url, params={**params, "apiKey": key}, timeout=20)
                resp.raise_for_status()
                data = resp.json()
                self.key_pool.record_use(key)
                if isinstance(data, dict) and data.get("error"):
                    print(f"[oddspapi] API error for {path}: {data['error']}")
                    continue
                cache_file.write_text(json.dumps({
                    "_fetched": datetime.now().isoformat(),
                    "data": data,
                }))
                return data
            except requests.RequestException as exc:
                print(f"[oddspapi] request failed {path}: {exc}")
                return None
        return None

    # ------------------------------------------------------------------ #
    # Endpoints
    # ------------------------------------------------------------------ #
    def tournaments(self, sport_id: int = 10) -> List[Dict[str, Any]]:
        data = self._get("v4/tournaments", {"sportId": sport_id},
                         cache_key=f"tournaments_{sport_id}", ttl_hours=24)
        return data if isinstance(data, list) else []

    def fixtures(self, tournament_ids: List[int]) -> List[Dict[str, Any]]:
        data = self._get("v4/fixtures", {"tournamentIds": ",".join(map(str, tournament_ids))},
                         cache_key=f"fixtures_{'_'.join(map(str, tournament_ids))}", ttl_hours=3)
        return data if isinstance(data, list) else []

    def odds(self, fixture_ids: List[str], bookmaker: str = "pinnacle") -> List[Dict[str, Any]]:
        data = self._get("v4/odds", {"fixtureIds": ",".join(fixture_ids), "bookmaker": bookmaker},
                         cache_key=f"odds_{bookmaker}_{'_'.join(fixture_ids)}", ttl_hours=3)
        return data if isinstance(data, list) else []

    def odds_by_tournaments(self, tournament_ids: List[int], bookmaker: str = "pinnacle") -> List[Dict[str, Any]]:
        data = self._get("v4/odds-by-tournaments",
                         {"tournamentIds": ",".join(map(str, tournament_ids)), "bookmaker": bookmaker},
                         cache_key=f"odds_by_tournaments_{bookmaker}_{'_'.join(map(str, tournament_ids))}",
                         ttl_hours=3)
        return data if isinstance(data, list) else []

    def historical_odds(self, fixture_id: str, bookmaker: str = "pinnacle") -> List[Dict[str, Any]]:
        """Historical odds with line-movement for a fixture (FREE on OddsPapi)."""
        data = self._get("v4/historical-odds",
                         {"fixtureId": fixture_id, "bookmaker": bookmaker},
                         cache_key=f"hist_odds_{bookmaker}_{fixture_id}", ttl_hours=24 * 7)
        return data if isinstance(data, list) else []

    def settlements(self, fixture_ids: List[str]) -> List[Dict[str, Any]]:
        data = self._get("v4/settlements", {"fixtureIds": ",".join(fixture_ids)},
                         cache_key=f"settlements_{'_'.join(fixture_ids)}", ttl_hours=24 * 7)
        return data if isinstance(data, list) else []

    def markets(self) -> List[Dict[str, Any]]:
        data = self._get("v4/markets", {}, cache_key="markets", ttl_hours=24 * 30)
        if isinstance(data, list):
            for m in data:
                mid = str(m.get("marketId") or "")
                name = m.get("marketName") or ""
                if mid and name:
                    self._market_names[mid] = name
                outcomes = {}
                for o in m.get("outcomes") or []:
                    oid = str(o.get("outcomeId") or "")
                    oname = o.get("outcomeName") or ""
                    if oid and oname:
                        outcomes[oid] = oname
                self._market_defs[mid] = {
                    "marketType": m.get("marketType") or "",
                    "outcomes": outcomes,
                    "handicap": m.get("handicap"),
                }
        return data if isinstance(data, list) else []

    def market_name(self, market_id: str) -> str:
        return self._market_names.get(str(market_id), str(market_id))


# ---------------------------------------------------------------------- #
# Parsing helpers
# ---------------------------------------------------------------------- #
def parse_oddspapi_odds(items: List[Dict[str, Any]],
                        market_defs: Dict[str, Dict[str, Any]] | None = None) -> Dict[str, Dict[str, float]]:
    """Flatten OddsPapi odds into {fixture_id: {market_key: best_price}}.

    Market keys use our internal naming (home_win, draw, away_win,
    over_2.5, under_2.5, ...). For each fixture we keep the best price
    across bookmakers. ``market_defs`` comes from ``OddsPapiClient.markets()``
    and maps marketId -> {marketType, outcomes: {outcomeId: outcomeName}}.
    """
    defs = market_defs or {}
    out: Dict[str, Dict[str, float]] = {}
    for item in items:
        fid = str(item.get("fixtureId", ""))
        if not fid:
            continue
        match_odds: Dict[str, float] = {}
        for bm, bm_odds in (item.get("bookmakerOdds") or {}).items():
            for mid, market in (bm_odds.get("markets") or {}).items():
                mdef = defs.get(str(mid), {})
                mtype = mdef.get("marketType", "")
                outcomes_map = mdef.get("outcomes", {})
                for oid, outcome in (market.get("outcomes") or {}).items():
                    players = outcome.get("players") or {}
                    if "0" not in players:
                        continue
                    price = float(players["0"].get("price", 0.0) or 0.0)
                    if price <= 1.0:
                        continue
                    oname = outcomes_map.get(str(oid), "")
                    key = _map_market(mtype, oname, players["0"], mdef)
                    if key and (key not in match_odds or price < match_odds[key]):
                        match_odds[key] = price
        if match_odds:
            out[fid] = match_odds
    return out


def _map_market(market_type: str, outcome_name: str, player: Dict[str, Any],
                mdef: Dict[str, Any]) -> Optional[str]:
    """Map an OddsPapi market type + outcome name to our internal key."""
    mt = (market_type or "").lower()
    oname = (outcome_name or "").lower()

    if mt in ("1x2", "moneyline", "full_time_result", "match_winner"):
        if oname in ("1", "home"):
            return "home_win"
        if oname in ("x", "draw"):
            return "draw"
        if oname in ("2", "away"):
            return "away_win"
    if "total" in mt or "over_under" in mt or "goals" in mt:
        point = player.get("point") or mdef.get("handicap")
        if point is not None:
            return f"{'over' if 'over' in oname else 'under'}_{float(point):.1f}"
    if "double_chance" in mt:
        if oname in ("1x", "home_or_draw"):
            return "home_or_draw"
        if oname in ("x2", "away_or_draw"):
            return "away_or_draw"
        if oname in ("12", "home_or_away"):
            return "home_or_away"
    if "both_to_score" in mt or "btts" in mt:
        if oname in ("yes", "gg"):
            return "btts_yes"
        if oname in ("no", "ng"):
            return "btts_no"
    return None


