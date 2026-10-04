"""Universal market scanner.

Evaluates every available market selection for a fixture and returns the
single highest-confidence, lowest-risk entry. Markets that cannot be
computed from the current workers (e.g. player props without lineup data)
are marked unavailable and excluded.
"""
from __future__ import annotations

from typing import Any, Dict, List

from src.markets.registry import MARKET_CATEGORIES


class UniversalMarketScanner:
    """Scans all market categories and picks the best selection."""

    def __init__(self, min_edge: float = 0.02):
        self.min_edge = min_edge

    def scan(self, fixture_label: str, probs: Dict[str, float],
             corners: Dict[str, float] | None = None,
             cards: Dict[str, float] | None = None,
             odds: Dict[str, float] | None = None) -> Dict[str, Any]:
        """Evaluate all markets for one fixture.

        ``probs``: goal-market probabilities from the Poisson matrix.
        ``corners``/``cards``: optional dicts of {market_key: probability}.
        Returns the single best selection or a SKIP decision.
        """
        candidates: List[Dict[str, Any]] = []

        # ---- Goal & outcome markets (from Poisson matrix) ----
        goal_specs = [
            ("home_over_0_5", probs.get("home_over_0_5", 0.0), "Home Team Over 0.5 Goals"),
            ("away_over_0_5", probs.get("away_over_0_5", 0.0), "Away Team Over 0.5 Goals"),
            ("over_0_5", probs.get("over_0_5", 0.0), "Match Over 0.5 Goals"),
            ("home_or_draw", probs.get("home_or_draw", 0.0), "DC: 1X (Home/Draw)"),
            ("away_or_draw", probs.get("away_or_draw", 0.0), "DC: X2 (Draw/Away)"),
            ("over_1_5", probs.get("over_1_5", 0.0), "OU Goals: Over 1.5"),
            ("over_2_5", probs.get("over_2_5", 0.0), "OU Goals: Over 2.5"),
            ("under_2_5", probs.get("under_2_5", 0.0), "OU Goals: Under 2.5"),
            ("under_3_5", probs.get("under_3_5", 0.0), "OU Goals: Under 3.5"),
            ("under_4_5", probs.get("under_4_5", 0.0), "OU Goals: Under 4.5"),
            ("home_win", probs.get("home_win", 0.0), "1X2: Home Win"),
            ("away_win", probs.get("away_win", 0.0), "1X2: Away Win"),
            ("btts_yes", probs.get("btts_yes", 0.0), "BTTS: Yes (GG)"),
            ("btts_no", probs.get("btts_no", 0.0), "BTTS: No (NG)"),
            ("home_clean_sheet", probs.get("home_clean_sheet", 0.0), "Home Clean Sheet: Yes"),
            ("away_clean_sheet", probs.get("away_clean_sheet", 0.0), "Away Clean Sheet: Yes"),
        ]
        for key, prob, label in goal_specs:
            self._add_candidate(candidates, key, prob, label, odds)

        # ---- Corner markets ----
        if corners:
            for key, prob in corners.items():
                self._add_candidate(candidates, key, prob, f"Corners: {key}", odds)

        # ---- Card markets ----
        if cards:
            for key, prob in cards.items():
                self._add_candidate(candidates, key, prob, f"Cards: {key}", odds)

        if not candidates:
            return {"fixture": fixture_label, "decision": "SKIP",
                    "reason": "no market cleared edge filter"}

        candidates.sort(key=lambda c: c["prob"], reverse=True)
        best = candidates[0]
        return {
            "fixture": fixture_label,
            "decision": "EXECUTE" if best["prob"] >= 0.85 else "SKIP",
            "market": best["label"],
            "key": best["key"],
            "confidence": round(best["prob"], 4),
            "odds": best.get("odds"),
            "edge": round(best.get("edge", 0.0), 4),
            "evaluated_markets": len(candidates),
        }

    def _add_candidate(self, candidates: List[Dict[str, Any]], key: str, prob: float,
                       label: str, odds: Dict[str, float] | None) -> None:
        if prob <= 0.0 or prob >= 1.0:
            return
        edge = 0.0
        if odds and key in odds and odds[key] > 1.0:
            edge = prob - 1.0 / odds[key]
            if edge < self.min_edge:
                return
        candidates.append({
            "key": key,
            "label": label,
            "prob": float(prob),
            "edge": edge,
            "odds": odds.get(key) if odds else None,
        })
