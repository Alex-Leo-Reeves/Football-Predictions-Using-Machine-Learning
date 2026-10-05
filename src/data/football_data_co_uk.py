"""football-data.co.uk CSV loader.

Reads locally-downloaded historical CSV files (results + odds + match stats)
from ``data/raw/football-data-co-uk/``. The site blocks automated scraping
(explicitly "NOT for bots/scrapers/AI"), so files must be downloaded manually
(free for personal use) and dropped into that directory.

Provides corners / cards / shots for training features — data the free
football-data.org tier does NOT include. No xG in these files (xG features
default to 0.0 unless another source is added).

CSV columns (see https://www.football-data.co.uk/notes.txt):
  Div, Date(dd/mm/yy), Time, HomeTeam, AwayTeam, FTHG, FTAG, FTR,
  HS, AS, HST, AST, HC, AC, HF, AF, HY, AY, HR, AR,
  B365H/D/A, PSH/D/A (Pinnacle), MaxH/D/A, AvgH/D/A, ...
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from src.config import get
from src.data.schemas import MatchResult


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _parse_date(value: Any) -> datetime:
    """Parse dd/mm/yy (or dd/mm/yyyy) into a naive datetime."""
    try:
        return datetime.strptime(str(value).strip(), "%d/%m/%y")
    except ValueError:
        try:
            return datetime.strptime(str(value).strip(), "%d/%m/%Y")
        except ValueError:
            return datetime(2000, 1, 1)


def load_football_data_co_uk_history(data_dir: str | None = None) -> List[MatchResult]:
    """Load all CSVs in the football-data.co.uk directory into MatchResults.

    Returns [] if the directory is empty/missing (caller falls back).
    """
    data_dir = Path(data_dir or get("data.football_data_co_uk.dir",
                                    "data/raw/football-data-co-uk"))
    if not data_dir.exists():
        return []

    results: List[MatchResult] = []
    seen: set = set()
    for csv_file in sorted(data_dir.glob("*.csv")):
        try:
            df = pd.read_csv(csv_file)
        except Exception as exc:  # pragma: no cover
            print(f"[football-data.co.uk] skipped {csv_file.name}: {exc}")
            continue
        for _, row in df.iterrows():
            home = str(row.get("HomeTeam", "")).strip()
            away = str(row.get("AwayTeam", "")).strip()
            fthg = row.get("FTHG")
            ftag = row.get("FTAG")
            if not home or not away or fthg is None or ftag is None:
                continue
            # Stable match id from date + teams (no API fixture id in CSVs).
            date_str = str(row.get("Date", "")).strip()
            mid = f"{date_str}|{home}|{away}"
            if mid in seen:
                continue
            seen.add(mid)
            results.append(MatchResult(
                match_id=mid,
                kickoff=_parse_date(date_str),
                home_team_id=home,
                away_team_id=away,
                home_goals=_to_int(fthg),
                away_goals=_to_int(ftag),
                home_xg=0.0,
                away_xg=0.0,
                # corners + cards (yellow + red) — the free stats this source adds
                home_corners=_to_int(row.get("HC")),
                away_corners=_to_int(row.get("AC")),
                home_cards=_to_int(row.get("HY")) + _to_int(row.get("HR")),
                away_cards=_to_int(row.get("AY")) + _to_int(row.get("AR")),
            ))
    return results
