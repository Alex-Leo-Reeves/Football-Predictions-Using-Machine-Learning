"""Game-state normalization.

A team leading 3-0 at minute 60 often eases off, deflating its stats.
This module re-weights attacking metrics based on the live scoreline so
that "performance while the game is competitive" carries more weight —
a key unpriced signal standard platforms miss.
"""
from __future__ import annotations

from typing import Dict


def game_state_weight(goal_diff: int, minute: float = 90.0) -> float:
    """Weight for attacking metrics given current goal difference.

    * tied (0): full weight 1.0
    * leading by 1: 0.85 (slight game management)
    * leading by 2+: 0.6 (heavy game management)
    * chasing by 1: 1.1 (pushing for an equaliser)
    * chasing by 2+: 1.2 (over-committed attacking shape)
    """
    if goal_diff == 0:
        return 1.0
    if goal_diff == 1:
        return 0.85
    if goal_diff >= 2:
        return 0.6
    if goal_diff == -1:
        return 1.1
    return 1.2


def normalize_npxg(raw_npxg: float, goal_diff: int, minute: float = 90.0) -> float:
    """Scale raw npxG by the inverse of game-state weight so that
    competitive-state performance is comparable across matches."""
    weight = game_state_weight(goal_diff, minute)
    # If a team was managing a lead (weight < 1), its raw npxG understates
    # true ability -> scale up. If chasing (weight > 1), scale down.
    return raw_npxg / weight if weight > 0 else raw_npxg


def state_adjustment_map() -> Dict[str, float]:
    return {
        "tied": game_state_weight(0),
        "lead_1": game_state_weight(1),
        "lead_2plus": game_state_weight(2),
        "chase_1": game_state_weight(-1),
        "chase_2plus": game_state_weight(-2),
    }
