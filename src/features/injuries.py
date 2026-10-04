"""Key Player Availability Index (KPAI) & lineup impact.

KPAI measures how much of a team's attacking/creative/defensive output
is available. It converts "when Player X plays they never lose" style
rules into a numeric 0..1 availability fraction used as a model feature.
"""
from __future__ import annotations

from typing import Dict, List

from src.config import get
from src.data.schemas import MatchResult, Player


def player_importance(player: Player) -> float:
    """Composite importance score for a player (used to rank key players)."""
    return (
        player.xg_per90 * 1.0
        + player.assists_per90 * 1.0
        + player.progressive_passes_per90 * 0.3
        + player.defensive_actions_per90 * 0.2
    )


def compute_kpai(players: List[Player], available_ids: List[str] | None,
                 top_n: int | None = None) -> float:
    """Availability fraction weighted by player importance.

    Returns 1.0 when all key players are available, lower when key
    players are missing. ``available_ids=None`` means full availability.
    """
    top_n = top_n or int(get("features.kpai.top_n", 3))
    if not players:
        return 1.0
    ranked = sorted(players, key=player_importance, reverse=True)
    key_players = ranked[:top_n]
    if available_ids is None:
        return 1.0
    available = set(available_ids)
    present = sum(1 for p in key_players if p.id in available)
    return present / len(key_players)


def kpai_from_missing(players: List[Player], missing_ids: List[str],
                      top_n: int | None = None) -> float:
    """KPAI given a list of missing (injured/suspended) player ids."""
    available = [p.id for p in players if p.id not in set(missing_ids)]
    return compute_kpai(players, available, top_n)
