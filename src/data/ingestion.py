"""Data ingestion.

Unified interface that uses real API sources (The Odds API, API-Football)
when keys are present, and falls back to the synthetic generator when they
are missing — so the pipeline runs end-to-end locally and on GitHub Actions.
"""
from __future__ import annotations

import os
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from src.config import get
from src.data.schemas import Fixture, MatchResult, naive
from src.data.synthetic import SyntheticLeague


class IngestionError(RuntimeError):
    pass


def _has_key(name: str) -> bool:
    """True if a key is present in env (singular OR plural _KEYS) or config.

    ``name`` is the singular form (e.g. ``FOOTBALL_API_KEY``); the plural
    secret is ``FOOTBALL_API_KEYS`` (replace trailing KEY with KEYS). Also
    checks the hardcoded ``data.api_keys`` config section.
    """
    plural = name[:-3] + "KEYS" if name.endswith("KEY") else name + "_KEYS"
    cfg_key = name.lower().replace("_key", "")
    return bool(
        os.getenv(name)
        or os.getenv(plural)
        or get(f"data.{name.lower()}", None)
        or get(f"data.{name.lower()}_keys", None)
        or get(f"data.api_keys.{cfg_key}", [])
    )


def _to_float(value: Any, default: float = 0.0) -> float:
    """Safely convert API values (may be str or None) to float."""
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_int(value: Any, default: int = 0) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------- #
# Real-data builders (API-Football + The Odds API)
# ---------------------------------------------------------------------- #
def build_real_history(client, league: int, seasons,
                       max_stats_fixtures: int = 60) -> List[MatchResult]:
    """Build MatchResult objects from API-Football across multiple seasons.

    Fetches all fixtures for each season (1 request each), then enriches up
    to ``max_stats_fixtures`` of the most recent completed matches with
    per-match statistics (xG, corners, cards) — budget-aware. ``seasons``
    may be a single int or an iterable of ints.
    """
    from src.data.api_football import parse_fixture_basic, parse_statistics

    if isinstance(seasons, int):
        seasons = [seasons]

    all_items: List[Dict[str, Any]] = []
    for season in seasons:
        items = client.fixtures_by_season(league, season)
        if items:
            all_items.extend(items)

    completed = [i for i in all_items if parse_fixture_basic(i).get("status") in ("FT", "AET", "PEN")]
    completed.sort(key=lambda i: parse_fixture_basic(i).get("kickoff") or "")

    # Enrich only the most recent N completed matches (budget-aware)
    to_enrich = completed[-max_stats_fixtures:]
    stats_cache: Dict[int, Dict[str, Any]] = {}
    for item in to_enrich:
        fid = parse_fixture_basic(item).get("fixture_id")
        if fid is None:
            continue
        raw_stats = client.fixture_statistics(fid)
        stats_cache[fid] = parse_statistics(raw_stats)

    results: List[MatchResult] = []
    seen: set = set()
    for item in completed:
        b = parse_fixture_basic(item)
        fid = b.get("fixture_id")
        if fid is None or b.get("home_goals") is None or b.get("away_goals") is None:
            continue
        if fid in seen:
            continue
        seen.add(fid)
        stats = stats_cache.get(fid, {"home": {}, "away": {}})
        results.append(MatchResult(
            match_id=str(fid),
            kickoff=datetime.fromisoformat(b["kickoff"].replace("Z", "+00:00")),
            home_team_id=str(b["home_id"]),
            away_team_id=str(b["away_id"]),
            home_goals=_to_int(b["home_goals"]),
            away_goals=_to_int(b["away_goals"]),
            home_xg=_to_float(stats["home"].get("expected_goals")),
            away_xg=_to_float(stats["away"].get("expected_goals")),
            home_corners=_to_int(stats["home"].get("Corner Kicks")),
            away_corners=_to_int(stats["away"].get("Corner Kicks")),
            home_cards=_to_int(stats["home"].get("Yellow Cards")),
            away_cards=_to_int(stats["away"].get("Yellow Cards")),
            home_npxg_state_adj=_to_float(stats["home"].get("expected_goals")),
            away_npxg_state_adj=_to_float(stats["away"].get("expected_goals")),
        ))
    return results


def build_real_today_fixtures(client, odds_client, oddspapi_client,
                              league: int, season: int,
                              league_ids: List[int] | None = None) -> List[Fixture]:
    """Build today's Fixture objects with odds attached (if available).

    Scans EVERY league in ``league_ids`` (defaults to ``[league]``) so the
    engine covers matches worldwide, not just one competition.

    Odds source priority: OddsPapi (primary, has historical + more
    bookmakers) -> The Odds API (fallback).
    """
    from src.data.api_football import parse_fixture_basic

    ids = league_ids or [league]
    # Cap the number of leagues scanned per run — API-Football's free plan
    # rate-limits at ~10 req/min, so scanning 281 leagues in one burst gets
    # 429'd. OddsPapi (the primary source) already covers ALL leagues.
    max_leagues = int(get("data.football_api.max_leagues_per_run", 30))
    ids = ids[:max_leagues]
    items: List[Dict[str, Any]] = []
    for lid in ids:
        items.extend(client.today_fixtures(lid, season))
    if not items:
        return []

    # Attach odds by matching team names
    odds_by_match: Dict[str, Dict[str, float]] = {}

    # 1) OddsPapi (primary) — match by team names via fixtures lookup
    if oddspapi_client is not None:
        try:
            from src.data.oddspapi import parse_oddspapi_odds
            oddspapi_client.markets()  # enrich market-id -> name mapping
            # Map API-Football team names to OddsPapi fixtures
            odds_by_match = _oddspapi_odds_by_team_names(oddspapi_client, items)
        except Exception as exc:  # pragma: no cover
            print(f"[ingestion] OddsPapi unavailable: {exc}")

    # 2) The Odds API (fallback)
    if not odds_by_match and odds_client is not None:
        from src.data.odds_api import parse_odds_payload
        sport = get("data.odds_api.sport", "soccer_epl")
        regions = get("data.odds_api.regions", "eu")
        odds_by_match = parse_odds_payload(odds_client.odds(sport, regions))

    fixtures: List[Fixture] = []
    for item in items:
        b = parse_fixture_basic(item)
        if b.get("fixture_id") is None:
            continue
        match_key = f"{b['home_name'].lower()} @ {b['away_name'].lower()}"
        fixtures.append(Fixture(
            fixture_id=str(b["fixture_id"]),
            kickoff=datetime.fromisoformat(b["kickoff"].replace("Z", "+00:00")),
            home_team_id=str(b["home_id"]),
            away_team_id=str(b["away_id"]),
            home_team_name=b["home_name"],
            away_team_name=b["away_name"],
            league=b["league_name"] or "",
            odds=odds_by_match.get(match_key, {}),
        ))
    return fixtures


def build_oddspapi_today_fixtures(oddspapi_client) -> List[Fixture]:
    """Build today's Fixture objects from OddsPapi (current season).

    Used when API-Football's free plan can't access the current season.
    OddsPapi fixtures carry participant names + start times; odds are
    attached from the same source.
    """
    from datetime import date, timedelta
    from src.data.oddspapi import parse_oddspapi_odds

    oddspapi_client.markets()  # enrich market definitions
    tournaments = oddspapi_client.tournaments(sport_id=10)
    tids = [t.get("tournamentId") for t in tournaments if t.get("tournamentId")]
    if not tids:
        return []

    # Fetch fixtures for ALL tournaments (chunked so the URL stays sane).
    # This is what makes the engine scan every league/competition today,
    # not just the first 20. Chunk size 20 keeps the URL within API limits.
    fixtures: List[Dict[str, Any]] = []
    for i in range(0, len(tids), 20):
        chunk = tids[i:i + 20]
        fixtures.extend(oddspapi_client.fixtures(chunk))
    today = date.today()
    today_start = datetime.combine(today, datetime.min.time())
    today_end = today_start + timedelta(days=1)

    # Filter to today's fixtures
    todays = []
    for f in fixtures:
        st = f.get("startTime")
        if not st:
            continue
        try:
            kickoff = datetime.fromisoformat(st.replace("Z", "+00:00"))
        except ValueError:
            continue
        if today_start <= naive(kickoff) < today_end:
            todays.append(f)

    # Fetch odds for today's fixtures
    fids = [str(f.get("fixtureId")) for f in todays if f.get("fixtureId")]
    odds_by_fid: Dict[str, Dict[str, float]] = {}
    if fids:
        odds_items = oddspapi_client.odds(fids[:50], bookmaker="pinnacle")
        odds_by_fid = parse_oddspapi_odds(odds_items, oddspapi_client._market_defs)

    out: List[Fixture] = []
    for f in todays:
        fid = str(f.get("fixtureId", ""))
        if not fid:
            continue
        kickoff = datetime.fromisoformat(str(f["startTime"]).replace("Z", "+00:00"))
        out.append(Fixture(
            fixture_id=fid,
            kickoff=kickoff,
            home_team_id=str(f.get("participant1Id", "")),
            away_team_id=str(f.get("participant2Id", "")),
            home_team_name=f.get("participant1Name", ""),
            away_team_name=f.get("participant2Name", ""),
            league=f.get("tournamentName", ""),
            odds=odds_by_fid.get(fid, {}),
        ))
    return out



    """Fetch OddsPapi odds and key them by 'home @ away' team names.

    OddsPapi uses its own fixture ids, so we map to API-Football fixtures
    by team names. Best-effort: if participant names aren't available in
    the response, returns {} and The Odds API fallback takes over.
    """
    from src.data.api_football import parse_fixture_basic
    from src.data.oddspapi import parse_oddspapi_odds

    # Build a name -> match_key map from API-Football today's fixtures
    name_to_key: Dict[str, str] = {}
    for item in api_football_items:
        b = parse_fixture_basic(item)
        if b.get("home_name") and b.get("away_name"):
            name_to_key[f"{b['home_name'].lower()} @ {b['away_name'].lower()}"] = str(b["fixture_id"])

    # Fetch odds for soccer tournaments (sportId=10)
    tournaments = oddspapi_client.tournaments(sport_id=10)
    tids = [t.get("tournamentId") for t in tournaments if t.get("tournamentId")]
    if not tids:
        return {}
    odds_items = oddspapi_client.odds_by_tournaments(tids, bookmaker="pinnacle")
    parsed = parse_oddspapi_odds(odds_items, oddspapi_client._market_names)

    # Build fixtureId -> (home, away) names from OddsPapi fixtures (all tids)
    fixtures: List[Dict[str, Any]] = []
    for i in range(0, len(tids), 20):
        fixtures.extend(oddspapi_client.fixtures(tids[i:i + 20]))
    fid_to_names: Dict[str, tuple] = {}
    for f in fixtures:
        fid = str(f.get("fixtureId", ""))
        if not fid:
            continue
        home = (f.get("participant1Name") or f.get("homeTeam") or
                f.get("participants", [{}])[0].get("name") if isinstance(f.get("participants"), list) and f.get("participants") else None)
        away = (f.get("participant2Name") or f.get("awayTeam") or
                f.get("participants", [{}])[1].get("name") if isinstance(f.get("participants"), list) and len(f.get("participants", [])) > 1 else None)
        if home and away:
            fid_to_names[fid] = (str(home).lower(), str(away).lower())

    # Map odds to match keys
    out: Dict[str, Dict[str, float]] = {}
    for fid, odds in parsed.items():
        names = fid_to_names.get(fid)
        if not names:
            continue
        match_key = f"{names[0]} @ {names[1]}"
        if match_key in name_to_key:
            out[match_key] = odds
    return out




# ---------------------------------------------------------------------- #
# football-data.org builders (free tier)
# ---------------------------------------------------------------------- #
def build_football_data_history(client, competitions, seasons) -> List[MatchResult]:
    """Build MatchResult objects from football-data.org (free tier).

    Free tier has no xG/corners/cards, so those default to 0.0.
    ``competitions`` / ``seasons`` may be a single value or an iterable.
    """
    from .football_data import parse_football_data_match

    if isinstance(competitions, str):
        competitions = [competitions]
    if isinstance(seasons, int):
        seasons = [seasons]

    results: List[MatchResult] = []
    seen: set = set()
    for code in competitions:
        for season in seasons:
            for item in client.matches_by_competition(code, season):
                b = parse_football_data_match(item)
                if b.get("status") not in ("FINISHED", "AWARDED"):
                    continue
                fid = b.get("fixture_id")
                if fid is None or b.get("home_goals") is None or b.get("away_goals") is None:
                    continue
                if fid in seen:
                    continue
                seen.add(fid)
                results.append(MatchResult(
                    match_id=str(fid),
                    kickoff=datetime.fromisoformat(b["kickoff"].replace("Z", "+00:00")),
                    home_team_id=str(b["home_id"]),
                    away_team_id=str(b["away_id"]),
                    home_goals=_to_int(b["home_goals"]),
                    away_goals=_to_int(b["away_goals"]),
                    home_xg=0.0,
                    away_xg=0.0,
                    home_corners=0,
                    away_corners=0,
                    home_cards=0,
                    away_cards=0,
                ))
    return results


def build_football_data_today_fixtures(client) -> List[Fixture]:
    """Build today's Fixture objects from football-data.org (free tier)."""
    from .football_data import parse_football_data_match

    today = date.today().isoformat()
    fixtures: List[Fixture] = []
    for item in client.matches_by_date(today, today):
        b = parse_football_data_match(item)
        if b.get("status") not in ("SCHEDULED", "TIMED"):
            continue
        if b.get("fixture_id") is None:
            continue
        fixtures.append(Fixture(
            fixture_id=str(b["fixture_id"]),
            kickoff=datetime.fromisoformat(b["kickoff"].replace("Z", "+00:00")),
            home_team_id=str(b["home_id"]),
            away_team_id=str(b["away_id"]),
            home_team_name=b["home_name"] or "",
            away_team_name=b["away_name"] or "",
            league=b["league"] or "",
            country=b["country"] or "",
            sport="football",
        ))
    return fixtures


def build_football_data_co_uk_history() -> List[MatchResult]:
    """Load locally-downloaded football-data.co.uk CSVs (corners/cards/shots)."""
    from .football_data_co_uk import load_football_data_co_uk_history
    return load_football_data_co_uk_history()


# ---------------------------------------------------------------------- #
# The Odds API builders (free tier — primary today-fixtures source)
# ---------------------------------------------------------------------- #
def build_odds_api_today_fixtures(odds_client) -> List[Fixture]:
    """Build today's Fixture objects directly from The Odds API.

    The Odds API returns upcoming games WITH odds in one call per sport key,
    so it replaces OddsPapi/API-Football as the primary source for today's
    fixtures. Free tier = 500 credits/month; each sport key = 1 credit.
    """
    from .odds_api import parse_odds_payload

    sport_keys = get("data.odds_api.sport_keys", ["soccer_epl"])
    regions = get("data.odds_api.regions", "eu")
    fixtures: List[Fixture] = []
    seen: set = set()
    for sport in sport_keys:
        items = odds_client.odds(sport, regions)
        odds_by_key = parse_odds_payload(items)
        for item in items:
            home = item.get("home_team", "")
            away = item.get("away_team", "")
            commence = item.get("commence_time")
            if not home or not away or not commence:
                continue
            fid = item.get("id") or f"{sport}_{home}_{away}"
            if fid in seen:
                continue
            seen.add(fid)
            match_key = f"{home.lower()} @ {away.lower()}"
            fixtures.append(Fixture(
                fixture_id=str(fid),
                kickoff=datetime.fromisoformat(commence.replace("Z", "+00:00")),
                home_team_id=home,
                away_team_id=away,
                home_team_name=home,
                away_team_name=away,
                league=sport,
                country="",
                sport="football",
                odds=odds_by_key.get(match_key, {}),
            ))
    return fixtures


# ---------------------------------------------------------------------- #
# Time-window filtering
# ---------------------------------------------------------------------- #
def filter_fixtures_by_time(fixtures: List[Fixture], now: datetime | None = None,
                            end_time: str = "23:30", days_ahead: int = 0) -> List[Fixture]:
    """Keep only fixtures that kick off between ``now`` and the window end.

    Matches that have already started (e.g. a 9am kickoff when the script
    runs at 10am) are excluded, as are matches after the betting window
    closes (default 23:30). ``end_time`` is a ``"HH:MM"`` 24h string.
    ``days_ahead`` extends the window N days beyond today (0 = today only),
    so the engine can also evaluate tomorrow's slate — the odds feed returns
    the next ~48h of games, and most of them kick off tomorrow.
    """
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)

    try:
        end_h, end_m = (int(x) for x in end_time.split(":"))
    except (ValueError, AttributeError):
        end_h, end_m = 23, 30

    window_end = (now + timedelta(days=days_ahead)).replace(
        hour=end_h, minute=end_m, second=0, microsecond=0)

    kept: List[Fixture] = []
    for fixture in fixtures:
        kickoff = fixture.kickoff
        if kickoff.tzinfo is None:
            kickoff = kickoff.replace(tzinfo=timezone.utc)
        if kickoff < now:
            continue  # already started / in the past
        if kickoff > window_end:
            continue  # after the betting window closes
        kept.append(fixture)
    return kept


# ---------------------------------------------------------------------- #
# Unified loader with mock fallback
# ---------------------------------------------------------------------- #
class DataLoader:
    """Loads historical matches + today's fixtures.

    With API keys present it pulls REAL data from API-Football and The
    Odds API (cache-aware, budget-aware). Without them it transparently
    falls back to the synthetic league so the pipeline stays runnable.
    """

    def __init__(self, use_mock: bool | None = None):
        self.use_mock = use_mock if use_mock is not None else bool(get("data.use_mock", True))
        self.synthetic: SyntheticLeague | None = None
        self.synthetic_basketball = None
        self._football_client = None
        self._odds_client = None
        self._oddspapi_client = None
        self._football_data_client = None

    def _ensure_synthetic(self) -> SyntheticLeague:
        if self.synthetic is None:
            self.synthetic = SyntheticLeague()
        return self.synthetic

    def _ensure_synthetic_basketball(self):
        """Lazily build the synthetic basketball league (no API keys needed)."""
        if self.synthetic_basketball is None:
            from src.data.synthetic_basketball import SyntheticBasketballLeague
            self.synthetic_basketball = SyntheticBasketballLeague()
        return self.synthetic_basketball

    def _ensure_real_clients(self):
        """Lazily build real API clients if keys are present."""
        if self._football_client is None and _has_key("FOOTBALL_API_KEY"):
            from src.data.api_football import APIFootballClient
            try:
                self._football_client = APIFootballClient()
            except RuntimeError:
                self._football_client = None
        if self._odds_client is None and _has_key("ODDS_API_KEY"):
            from src.data.odds_api import OddsAPIClient
            try:
                self._odds_client = OddsAPIClient()
            except RuntimeError:
                self._odds_client = None
        if self._oddspapi_client is None and _has_key("ODDSPAPI_KEY"):
            from src.data.oddspapi import OddsPapiClient
            try:
                self._oddspapi_client = OddsPapiClient()
            except RuntimeError:
                self._oddspapi_client = None
        if self._football_data_client is None and _has_key("FOOTBALL_DATA_ORG_KEY"):
            from src.data.football_data import FootballDataClient
            try:
                self._football_data_client = FootballDataClient()
            except RuntimeError:
                self._football_data_client = None
        return (self._football_client, self._odds_client,
                self._oddspapi_client, self._football_data_client)

    @property
    def has_real_data(self) -> bool:
        fb, _, _, fdata = self._ensure_real_clients()
        return fb is not None or fdata is not None

    def load_history(self) -> List[MatchResult]:
        """Return historical match results (real or synthetic)."""
        if self.use_mock:
            return self._ensure_synthetic().history
        fb, _, _, fdata = self._ensure_real_clients()
        # 1) football-data.co.uk CSVs (richest free source: corners/cards/shots)
        try:
            co_uk = build_football_data_co_uk_history()
            if co_uk:
                print(f"[ingestion] football-data.co.uk: {len(co_uk)} historical matches loaded")
                return co_uk
        except Exception as exc:  # pragma: no cover
            print(f"[ingestion] football-data.co.uk history failed: {exc}")
        # 2) football-data.org (free tier — results only, no xG/corners/cards)
        if fdata is not None:
            comps = get("data.football_data.competitions",
                        ["PL", "ELC", "BL1", "SA", "PD", "FL1", "DED", "PPL", "CL"])
            seasons = get("data.football_data.seasons", [2025, 2024])
            real = build_football_data_history(fdata, comps, seasons)
            if real:
                print(f"[ingestion] football-data.org: {len(real)} historical matches loaded")
                return real
            print("[ingestion] football-data.org returned no history")
        # 3) API-Football (legacy)
        if fb is not None:
            league = int(get("data.football_api.league_id", 39))
            seasons = get("data.football_api.seasons", [int(get("data.football_api.season", 2024))])
            max_stats = int(get("data.football_api.max_stats_fixtures", 60))
            real = build_real_history(fb, league, seasons, max_stats_fixtures=max_stats)
            if real:
                return real
            print("[ingestion] no real history returned — falling back to synthetic")
        return self._ensure_synthetic().history

    def load_today_fixtures(self) -> List[Fixture]:
        """Return today's fixtures (real with odds, or synthetic)."""
        if self.use_mock:
            fx = self._ensure_synthetic().today_fixtures()
            print(f"[ingestion] use_mock=True — using SYNTHETIC fixtures ({len(fx)})")
            return fx
        fb, odds, oddspapi, fdata = self._ensure_real_clients()
        print(f"[ingestion] real clients: football={fb is not None} odds={odds is not None} oddspapi={oddspapi is not None} fdata={fdata is not None}")
        # PRIMARY: The Odds API (free tier — returns fixtures WITH odds)
        if odds is not None:
            try:
                real = build_odds_api_today_fixtures(odds)
                if real:
                    print(f"[ingestion] The Odds API: {len(real)} real fixtures loaded")
                    return real
            except Exception as exc:  # pragma: no cover
                print(f"[ingestion] The Odds API today fixtures failed: {exc}")
        # OddsPapi (legacy primary)
        if oddspapi is not None:
            try:
                real = build_oddspapi_today_fixtures(oddspapi)
                if real:
                    print(f"[ingestion] OddsPapi: {len(real)} real fixtures loaded")
                    return real
            except Exception as exc:  # pragma: no cover
                print(f"[ingestion] OddsPapi today fixtures failed: {exc}")
        # API-Football (legacy)
        if fb is not None:
            league = int(get("data.football_api.league_id", 39))
            season = int(get("data.football_api.season", 2024))
            league_ids = get("data.football_api.league_ids", [league])
            real = build_real_today_fixtures(fb, odds, oddspapi, league, season,
                                             league_ids=league_ids)
            if real:
                print(f"[ingestion] API-Football: {len(real)} real fixtures loaded")
                return real
            print("[ingestion] API-Football returned no today fixtures")
        # football-data.org (free tier)
        if fdata is not None:
            try:
                real = build_football_data_today_fixtures(fdata)
                if real:
                    print(f"[ingestion] football-data.org: {len(real)} real fixtures loaded")
                    return real
            except Exception as exc:  # pragma: no cover
                print(f"[ingestion] football-data.org today fixtures failed: {exc}")
        fx = self._ensure_synthetic().today_fixtures()
        print(f"[ingestion] FALLING BACK to synthetic fixtures ({len(fx)})")
        return fx

    def load_basketball_history(self) -> List[MatchResult]:
        """Return basketball historical match results (synthetic for now)."""
        return self._ensure_synthetic_basketball().history

    def load_today_basketball_fixtures(self) -> List[Fixture]:
        """Return today's basketball fixtures (synthetic for now)."""
        return self._ensure_synthetic_basketball().today_fixtures()

    def basketball_teams(self) -> Dict[str, Any]:
        return self._ensure_synthetic_basketball().teams

    def teams(self) -> Dict[str, Any]:
        return self._ensure_synthetic().teams

    def players(self) -> Dict[str, Any]:
        return self._ensure_synthetic().players
