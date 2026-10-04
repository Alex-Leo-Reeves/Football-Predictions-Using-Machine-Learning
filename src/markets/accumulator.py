"""Accumulator builder.

Chains independent, high-confidence selections to reach a target combined
odds while enforcing:
  * one selection per fixture (no correlated same-match stacking)
  * no two selections sharing a team (cross-correlation guard)
  * cumulative probability reporting (honest decay math)
  * optional system-bet split suggestions

Target-odds optimizer
---------------------
The builder OVER-SELECTS (every qualified independent fixture is included)
and then, if the combined odds exceed the target (e.g. 450 vs 300), it brings
the ticket back down toward the target by:

  * downgrading a selection to a "leaner" variant (e.g. home_win -> double
    chance 1X -> double chance 12), which lowers its odds and raises its
    confidence, and/or
  * removing the weakest (lowest-confidence) legs.

This mirrors the requested behaviour: make the best prediction for every
match, then trim/downgrade until the ticket lands near 300 odds. If the
combined odds are below the target, the builder keeps everything — it never
forces an inaccurate pick just to hit 300.
"""
from __future__ import annotations

import itertools
import math
from typing import Any, Dict, List

from src.config import get


class AccumulatorBuilder:
    def __init__(self, target_odds: float | None = None, min_confidence: float | None = None,
                 max_legs: int | None = None, max_same_match_legs: int | None = None,
                 system_bets: bool | None = None, target_tolerance: float | None = None):
        self.target_odds = target_odds if target_odds is not None else float(get("pipeline.target_odds", 300.0))
        self.min_confidence = min_confidence if min_confidence is not None else float(
            get("pipeline.min_confidence", 0.90))
        self.max_legs = max_legs or int(get("pipeline.max_legs", 30))
        self.max_same_match_legs = max_same_match_legs or int(get("pipeline.max_same_match_legs", 1))
        # ALL-OR-NOTHING by default: one wrong leg = whole ticket loss.
        # System-bet splits hedge against that and are OFF unless enabled.
        self.system_bets = system_bets if system_bets is not None else bool(get("pipeline.system_bets", False))
        # How close to the target we need to land before stopping the
        # downgrade loop (fraction, e.g. 0.05 = ±5% of 300 = 285..315).
        self.target_tolerance = target_tolerance if target_tolerance is not None else float(
            get("pipeline.target_tolerance", 0.05))

    # ------------------------------------------------------------------ #
    # Independence checks
    # ------------------------------------------------------------------ #
    @staticmethod
    def _teams_of(selection: Dict[str, Any]) -> set:
        """Extract team ids involved in a selection."""
        teams = set()
        for key in ("home_id", "away_id"):
            if selection.get(key):
                teams.add(selection[key])
        return teams

    def _is_independent(self, selection: Dict[str, Any], ticket: List[Dict[str, Any]]) -> bool:
        """Reject if the selection shares a fixture or a team with ticket legs."""
        sel_teams = self._teams_of(selection)
        for leg in ticket:
            # same fixture -> correlated
            if leg.get("fixture_id") and leg["fixture_id"] == selection.get("fixture_id"):
                return False
            # shared team -> correlated (e.g. same team goals + opponent cards)
            if sel_teams and sel_teams & self._teams_of(leg):
                return False
        return True

    # ------------------------------------------------------------------ #
    # Ticket building
    # ------------------------------------------------------------------ #
    def build(self, selections: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Build the best ticket, over-selecting then optimizing to target.

        ``selections`` may carry a ``variants`` list (alternative markets
        for the same fixture with lower odds / higher confidence) used to
        downgrade legs when the combined odds overshoot the target.
        """
        # sort by confidence desc, then odds asc (prefer safe legs first)
        ranked = sorted(selections, key=lambda s: (-s.get("confidence", 0.0), s.get("odds", 99.0)))

        # ---- Step 1: over-select every qualified independent fixture ----
        ticket: List[Dict[str, Any]] = []
        for sel in ranked:
            if len(ticket) >= self.max_legs:
                break
            if sel.get("confidence", 0.0) < self.min_confidence:
                continue
            if not self._is_independent(sel, ticket):
                continue
            odds = float(sel.get("odds") or 1.0)
            if odds <= 1.0:
                continue
            ticket.append(sel)

        # ---- Step 2: optimize combined odds toward the target ----
        ticket, combined_odds, steps = self._optimize_to_target(ticket, self.target_odds)
        cumulative_prob = 1.0
        for sel in ticket:
            cumulative_prob *= float(sel.get("confidence", 0.0))

        return {
            "status": "TICKET_GENERATED" if ticket else "NO_QUALIFIED_SELECTIONS",
            "total_legs": len(ticket),
            "combined_odds": round(combined_odds, 2),
            "estimated_probability": round(cumulative_prob, 6),
            "target_odds": self.target_odds,
            # "Reached" means we landed inside the tolerance band around the
            # target (e.g. 285..315 for 300 @ 5%), not strictly >= 300.
            "target_reached": combined_odds >= self.target_odds * (1.0 - self.target_tolerance),
            "optimization_steps": steps,
            # ALL-OR-NOTHING: if ANY leg fails the whole ticket is a loss.
            "all_or_nothing": True,
            "system_bets": self._system_bet_suggestions(ticket) if self.system_bets else [],
            "selections": ticket,
        }

    # ------------------------------------------------------------------ #
    # Target-odds optimizer
    # ------------------------------------------------------------------ #
    def _optimize_to_target(self, ticket: List[Dict[str, Any]], target: float
                            ) -> tuple[List[Dict[str, Any]], float, List[Dict[str, Any]]]:
        """Bring combined odds down toward ``target`` when they overshoot.

        Returns ``(final_legs, combined_odds, steps)``. ``steps`` records
        every downgrade/removal/upgrade applied (for transparency).

        Strategy (safety-first, mirrors the user's use case):
          1. DOWNGRADE every leg to its leanest (lowest-odds, highest-
             confidence) variant — e.g. home_win -> double chance 1X ->
             double chance 12. This is the safest possible ticket.
          2. If still above the upper band, REMOVE the weakest legs.
          3. If that overshoots the lower band, selectively UPGRADE legs
             back up the odds ladder to land as close to the target as
             possible (never above the upper band).
        """
        if not ticket:
            return ticket, 1.0, []

        tol = max(0.02, self.target_tolerance)
        lower = target * (1.0 - tol)
        upper = target * (1.0 + tol)

        def product(legs: List[Dict[str, Any]]) -> float:
            p = 1.0
            for leg in legs:
                p *= float(leg.get("odds", 1.0))
            return p

        current = product(ticket)
        if current <= upper:
            # Already at or below target — keep everything (never force picks).
            return ticket, current, []

        steps: List[Dict[str, Any]] = []

        def ladder(leg: Dict[str, Any]) -> List[tuple]:
            """All (odds, market, key, confidence) levels for a leg, high->low."""
            levels = [(float(leg.get("odds", 1.0)), leg.get("market"),
                       leg.get("key"), leg.get("confidence", 0.0))]
            for v in leg.get("variants", []):
                vo = float(v.get("odds", 1.0))
                if vo < levels[0][0]:
                    levels.append((vo, v.get("market"), v.get("key"),
                                   v.get("confidence", 0.0)))
            levels.sort(key=lambda x: -x[0])
            return levels

        # Each entry tracks the original leg, its odds ladder and current level.
        entries = [{"leg": leg, "ladder": ladder(leg), "level": len(ladder(leg)) - 1}
                   for leg in ticket]

        def build_legs() -> List[Dict[str, Any]]:
            legs = []
            for e in entries:
                odds, market, key, conf = e["ladder"][e["level"]]
                legs.append({**e["leg"], "odds": odds, "market": market,
                             "key": key, "confidence": conf})
            return legs

        # ---- Phase 1: downgrade every leg to its leanest variant ----
        for i, e in enumerate(entries):
            if e["level"] < len(e["ladder"]) - 1:
                steps.append({
                    "type": "downgrade", "index": i,
                    "fixture_id": e["leg"].get("fixture_id"),
                    "from_market": e["leg"].get("market"),
                    "from_odds": e["leg"].get("odds"),
                    "to_market": e["ladder"][-1][1],
                    "to_odds": e["ladder"][-1][0],
                    "combined_odds_after": None,
                })

        legs = build_legs()
        current = product(legs)

        # ---- Phase 2: remove weakest legs while still above the upper band ----
        while current > upper and len(legs) > 1:
            idx = min(range(len(entries)),
                      key=lambda i: (entries[i]["leg"].get("confidence", 1.0),
                                     -float(entries[i]["leg"].get("odds", 1.0))))
            removed = entries.pop(idx)
            legs = build_legs()
            current = product(legs)
            steps.append({
                "type": "remove", "index": idx,
                "fixture_id": removed["leg"].get("fixture_id"),
                "from_market": removed["leg"].get("market"),
                "from_odds": removed["leg"].get("odds"),
                "to_market": None, "to_odds": None,
                "combined_odds_after": round(current, 2),
            })

        # ---- Phase 3: if below the lower band, upgrade legs to get closest ----
        # to the target without exceeding the upper band.
        guard = 0
        while current < lower and len(entries) > 1 and guard < 200:
            guard += 1
            best = None  # (score, entry_idx, new_level, new_product)
            for i, e in enumerate(entries):
                if e["level"] <= 0:
                    continue  # already at the highest-odds level
                new_level = e["level"] - 1
                new_odds = e["ladder"][new_level][0]
                new_product = current / e["ladder"][e["level"]][0] * new_odds
                score = abs(math.log(new_product) - math.log(target))
                if new_product > upper:
                    score += 0.05  # penalize overshooting the upper band
                if best is None or score < best[0]:
                    best = (score, i, new_level, new_product)
            if best is None:
                break
            _, i, new_level, new_product = best
            e = entries[i]
            steps.append({
                "type": "upgrade", "index": i,
                "fixture_id": e["leg"].get("fixture_id"),
                "from_market": e["ladder"][e["level"]][1],
                "from_odds": e["ladder"][e["level"]][0],
                "to_market": e["ladder"][new_level][1],
                "to_odds": e["ladder"][new_level][0],
                "combined_odds_after": round(new_product, 2),
            })
            e["level"] = new_level
            current = new_product
            legs = build_legs()

        # Fill in the combined odds for phase-1 downgrade steps.
        for s in steps:
            if s.get("combined_odds_after") is None:
                s["combined_odds_after"] = round(current, 2)

        return legs, current, steps

    # ------------------------------------------------------------------ #
    # System bet suggestions
    # ------------------------------------------------------------------ #
    @staticmethod
    def _system_bet_suggestions(ticket: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Suggest system-bet splits (e.g. 4/5, 5/6) to hedge single-leg risk."""
        n = len(ticket)
        if n < 3:
            return []
        suggestions = []
        for k in (n - 1, n - 2):
            if k >= 2:
                n_combos = math.comb(n, k)
                suggestions.append({
                    "type": f"{k}/{n} system",
                    "combinations": n_combos,
                    "note": f"pays out if at least {k} of {n} legs hit",
                })
        return suggestions


# ---------------------------------------------------------------------- #
# Settlement (ALL-OR-NOTHING)
# ---------------------------------------------------------------------- #
def resolve_market_outcome(match, market_key: str) -> bool:
    """Resolve whether a market selection hit, given the final MatchResult.

    Handles 1X2, double chance, BTTS, clean sheets, team totals and generic
    over/under totals (including fractional lines like ``over_2.8``).
    """
    k = (market_key or "").lower()
    if k == "home_win":
        return match.home_goals > match.away_goals
    if k == "away_win":
        return match.away_goals > match.home_goals
    if k == "draw":
        return match.home_goals == match.away_goals
    if k == "home_or_draw":
        return match.home_goals >= match.away_goals
    if k == "away_or_draw":
        return match.away_goals >= match.home_goals
    if k == "btts_yes":
        return match.home_goals > 0 and match.away_goals > 0
    if k == "btts_no":
        return match.home_goals == 0 or match.away_goals == 0
    if k == "home_clean_sheet":
        return match.away_goals == 0
    if k == "away_clean_sheet":
        return match.home_goals == 0
    if k == "home_over_0_5":
        return match.home_goals > 0
    if k == "away_over_0_5":
        return match.away_goals > 0
    if k.startswith("over_"):
        try:
            return match.total_goals > float(k.split("_", 1)[1])
        except ValueError:
            return False
    if k.startswith("under_"):
        try:
            return match.total_goals < float(k.split("_", 1)[1])
        except ValueError:
            return False
    return False


def settle_ticket(ticket: Dict[str, Any], results_by_fixture: Dict[str, Any],
                  resolve=None) -> Dict[str, Any]:
    """Settle an accumulator ticket ALL-OR-NOTHING.

    Standard accumulator rule: if ANY leg fails, the whole ticket is a loss
    (``status`` = "LOSS"). Only when EVERY leg hits is it a "WIN".

    ``results_by_fixture`` maps fixture_id -> MatchResult (final score).
    ``resolve`` is an optional callable ``(selection, MatchResult) -> bool``;
    defaults to :func:`resolve_market_outcome` keyed on the selection's
    ``key`` (falling back to ``market``).
    """
    if resolve is None:
        def resolve(sel, match):
            return resolve_market_outcome(match, sel.get("key") or sel.get("market") or "")

    selections = ticket.get("selections", []) if isinstance(ticket, dict) else ticket
    legs = []
    for sel in selections:
        fid = sel.get("fixture_id")
        match = results_by_fixture.get(fid)
        hit = bool(match and resolve(sel, match))
        legs.append({
            "fixture_id": fid,
            "market": sel.get("market", ""),
            "key": sel.get("key", ""),
            "hit": hit,
        })

    total = len(legs)
    hit_legs = sum(1 for leg in legs if leg["hit"])
    return {
        "status": "WIN" if total and hit_legs == total else "LOSS",
        "all_or_nothing": True,
        "hit_legs": hit_legs,
        "total_legs": total,
        "legs": legs,
    }
