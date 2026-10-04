"""Master Brain — the meta-controller / arbitrator.

Pipeline per fixture:
  1. Hard veto rules (derby, weather, motivation, lineup uncertainty)
  2. Worker predictions (goals -> Poisson matrix, corners, cards)
  3. Auditor checks (anomaly, variance, structural contradiction)
  4. LLM expert (DeepSeek) — only if fixture passes the statistical gate
  5. Intelligent mixing of statistical + LLM probabilities
  6. Conformal / threshold gate -> EXECUTE or SKIP
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from src.config import get
from src.data.schemas import Fixture
from src.features.builder import FEATURE_COLUMNS
from src.models.anomaly import AnomalyAuditor, VarianceAuditor
from src.models.conformal import ConformalClassifier
from src.models.workers import BasketballWorker, CountWorker, GoalWorker


class MasterBrain:
    def __init__(self, goal_worker: GoalWorker, corner_worker: CountWorker | None = None,
                 card_worker: CountWorker | None = None,
                 basketball_worker: BasketballWorker | None = None,
                 anomaly_auditor: AnomalyAuditor | None = None,
                 variance_auditor: VarianceAuditor | None = None,
                 conformal: ConformalClassifier | None = None,
                 llm_expert: Any | None = None,
                 llm_weight: float | None = None,
                 high_floor_threshold: float | None = None,
                 primary_threshold: float | None = None,
                 min_edge: float | None = None,
                 max_variants: int | None = None):
        self.goal_worker = goal_worker
        self.corner_worker = corner_worker
        self.card_worker = card_worker
        self.basketball_worker = basketball_worker
        self.anomaly_auditor = anomaly_auditor
        self.variance_auditor = variance_auditor or VarianceAuditor()
        self.conformal = conformal
        self.llm_expert = llm_expert
        self.llm_weight = llm_weight if llm_weight is not None else float(get("llm.mix_weight", 0.3))
        self.high_floor_threshold = high_floor_threshold if high_floor_threshold is not None else float(
            get("markets.thresholds.high_floor", 0.90))
        self.primary_threshold = primary_threshold if primary_threshold is not None else float(
            get("markets.thresholds.primary", 0.80))
        self.min_edge = min_edge if min_edge is not None else float(get("markets.min_edge", 0.02))
        self.max_variants = max_variants if max_variants is not None else int(
            get("pipeline.max_variants_per_fixture", 5))

    # ------------------------------------------------------------------ #
    # Hard veto rules
    # ------------------------------------------------------------------ #
    @staticmethod
    def _hard_veto(fixture: Fixture) -> List[str]:
        """Deterministic veto rules. Returns list of violated rules."""
        flags = []
        if "derby" in fixture.risk_flags:
            flags.append("derby_high_variance")
        if "weather" in fixture.risk_flags:
            flags.append("extreme_weather")
        if "motivation" in fixture.risk_flags:
            flags.append("motivation_anomaly")
        if fixture.home_kpai < 0.6 or fixture.away_kpai < 0.6:
            flags.append("lineup_disruption")
        return flags

    # ------------------------------------------------------------------ #
    # Mixing statistical + LLM probabilities
    # ------------------------------------------------------------------ #
    def _mix(self, stat_probs: Dict[str, float], llm_probs: Dict[str, float] | None) -> Dict[str, float]:
        """Weighted blend of statistical and LLM probabilities."""
        if llm_probs is None:
            return dict(stat_probs)
        w = self.llm_weight
        mixed = {}
        for key in stat_probs:
            llm_key = {
                "home_win": "home_win_prob",
                "draw": "draw_prob",
                "away_win": "away_win_prob",
                "home_over_0_5": "home_over_0_5",
                "away_over_0_5": "away_over_0_5",
                "over_1_5": "over_1_5",
                "over_2_5": "over_2_5",
                "btts_yes": "btts_yes",
            }.get(key)
            if llm_key and llm_key in llm_probs:
                mixed[key] = (1.0 - w) * stat_probs[key] + w * llm_probs[llm_key]
            else:
                mixed[key] = stat_probs[key]

    # ------------------------------------------------------------------ #
    # Full fixture evaluation
    # ------------------------------------------------------------------ #
    def evaluate_fixture(self, fixture: Fixture, features: Dict[str, float],
                         odds: Dict[str, float] | None = None,
                         recent_matches: Dict[str, Any] | None = None,
                         h2h: Dict[str, Any] | None = None,
                         injuries: Dict[str, Any] | None = None) -> Dict[str, Any]:
        """Evaluate one fixture and return the decision.

        ``recent_matches`` / ``h2h`` / ``injuries`` are optional RAW data
        blocks passed to the LLM expert (leverages the 350k context window).
        """
        # 1. Hard veto rules
        vetoes = self._hard_veto(fixture)
        if vetoes:
            return {"fixture": fixture.label, "decision": "SKIP", "reason": "; ".join(vetoes)}

        # 2. Worker predictions (sport-aware)
        import pandas as pd
        X = pd.DataFrame([features], columns=FEATURE_COLUMNS)
        if fixture.sport == "basketball":
            if self.basketball_worker is None:
                return {"fixture": fixture.label, "decision": "SKIP",
                        "reason": "no basketball worker trained"}
            stat_probs = self.basketball_worker.predict_probs(X)[0]
        else:
            stat_probs = self.goal_worker.predict_probs(X)[0]

        # 3. Auditor checks
        auditor_flags = self.variance_auditor.check(features)
        if self.anomaly_auditor is not None:
            feat_vec = np.array([features[c] for c in FEATURE_COLUMNS], dtype=float)
            if self.anomaly_auditor.is_outlier(feat_vec):
                auditor_flags.append("anomaly_outlier")
        if auditor_flags:
            return {"fixture": fixture.label, "decision": "SKIP",
                    "reason": "auditor: " + "; ".join(auditor_flags)}

        # 3b. Conformal risk gate (strict, alpha <= 0.001). If the conformal
        # prediction set for this fixture spans MORE than one outcome class,
        # the engine abstains — this is the "bulletproof or nothing" rule.
        if self.conformal is not None:
            try:
                pset, max_prob = self.conformal.predict_single(
                    np.array([features[c] for c in FEATURE_COLUMNS], dtype=float))
                if len(pset) > 1:
                    return {"fixture": fixture.label, "decision": "SKIP",
                            "reason": f"conformal gate: prediction set {pset} wider than 1 class"}
            except Exception as exc:  # pragma: no cover
                print(f"[master_brain] conformal gate error for {fixture.label}: {exc}")

        # 4. LLM expert — only if statistical gate passes (saves API cost)
        llm_probs = None
        llm_decision = None
        if self.llm_expert is not None and fixture.sport != "basketball":
            gate = max(stat_probs.get("home_over_0_5", 0.0), stat_probs.get("away_over_0_5", 0.0),
                       stat_probs.get("home_or_draw", 0.0), stat_probs.get("away_or_draw", 0.0))
            if gate >= self.primary_threshold:
                try:
                    llm_out = self.llm_expert.predict(
                        fixture.label, features,
                        recent_matches=recent_matches,
                        h2h=h2h,
                        injuries=injuries,
                        odds=odds,
                    )
                    llm_probs = llm_out
                    llm_decision = llm_out.get("decision", "SKIP")
                except Exception as exc:  # pragma: no cover - network/API issues
                    print(f"[master_brain] LLM expert failed for {fixture.label}: {exc}")
                    llm_probs = None

        # 5. Mix
        mixed = self._mix(stat_probs, llm_probs)

        # 6. Rank markets + apply edge filter (sport-aware)
        if fixture.sport == "basketball":
            candidates = self._rank_basketball_markets(mixed, odds)
        else:
            candidates = self._rank_markets(mixed, odds)
        if not candidates:
            return {"fixture": fixture.label, "decision": "SKIP",
                    "reason": "no market cleared edge filter"}

        best = candidates[0]
        market, prob = best["market"], best["prob"]

        # 7. Threshold gate
        threshold = self.high_floor_threshold if best["high_floor"] else self.primary_threshold
        if prob >= threshold:
            return {
                "fixture": fixture.label,
                "decision": "EXECUTE",
                "market": market,
                "key": best["key"],
                "confidence": round(prob, 4),
                "odds": best.get("odds"),
                "edge": round(best.get("edge", 0.0), 4),
                "reason": best.get("reason", ""),
                "llm_used": llm_probs is not None,
                "llm_decision": llm_decision,
                # Top-N ranked markets as downgrade variants for the
                # accumulator's target-odds optimizer.
                "variants": candidates[:self.max_variants],
            }
        return {"fixture": fixture.label, "decision": "SKIP",
                "reason": f"best {market} at {prob:.1%} below threshold {threshold:.1%}"}

    # ------------------------------------------------------------------ #
    # Market ranking
    # ------------------------------------------------------------------ #
    def _rank_markets(self, probs: Dict[str, float],
                      odds: Dict[str, float] | None) -> List[Dict[str, Any]]:
        """Rank candidate markets by probability, applying edge filter."""
        candidates: List[Dict[str, Any]] = []

        # (market_key, probability, high_floor, human label)
        specs = [
            ("home_over_0_5", probs["home_over_0_5"], True, "Home Team Over 0.5 Goals"),
            ("away_over_0_5", probs["away_over_0_5"], True, "Away Team Over 0.5 Goals"),
            ("over_0_5", probs["over_0_5"], True, "Match Over 0.5 Goals"),
            ("home_or_draw", probs["home_or_draw"], True, "Double Chance 1X (Home/Draw)"),
            ("away_or_draw", probs["away_or_draw"], True, "Double Chance X2 (Draw/Away)"),
            ("home_or_away", probs.get("home_or_away", 0.0), True, "Double Chance 12 (Home/Away)"),
            ("over_1_5", probs["over_1_5"], False, "Match Over 1.5 Goals"),
            ("over_2_5", probs["over_2_5"], False, "Match Over 2.5 Goals"),
            ("home_win", probs["home_win"], False, "Home Win"),
            ("away_win", probs["away_win"], False, "Away Win"),
            ("btts_yes", probs["btts_yes"], False, "Both Teams To Score"),
        ]

        for key, prob, high_floor, label in specs:
            edge = 0.0
            if odds and key in odds and odds[key] > 1.0:
                implied = 1.0 / odds[key]
                edge = prob - implied
                if edge < self.min_edge:
                    continue  # negative/insufficient value — reject
            candidates.append({
                "market": label,
                "key": key,
                "prob": float(prob),
                "high_floor": high_floor,
                "edge": edge,
                "odds": odds.get(key) if odds else None,
                "reason": f"{label} @ {prob:.1%}",
            })

        candidates.sort(key=lambda c: c["prob"], reverse=True)
        return candidates

    # ------------------------------------------------------------------ #
    # Market ranking (basketball)
    # ------------------------------------------------------------------ #
    def _rank_basketball_markets(self, probs: Dict[str, float],
                                 odds: Dict[str, float] | None) -> List[Dict[str, Any]]:
        """Rank basketball markets by probability, applying edge filter."""
        candidates: List[Dict[str, Any]] = []
        total_line = float(get("markets.basketball.total_line", 220.5))
        spread = float(get("markets.basketball.spread", -5.5))
        alt_spread = float(get("markets.basketball.alt_spread", -12.5))
        home_total = float(get("markets.basketball.home_total", 108.5))

        specs = [
            ("home_win", probs["home_win"], False, "BB Moneyline: Home Win"),
            ("away_win", probs["away_win"], False, "BB Moneyline: Away Win"),
            ("home_cover", probs["home_cover"], False, f"BB Spread: Home Cover ({spread:+.1f})"),
            ("away_cover", probs["away_cover"], False, f"BB Spread: Away Cover ({-spread:+.1f})"),
            ("home_cover", probs["home_cover"], False, f"BB Alternate Spread: Home Cover ({alt_spread:+.1f})"),
            ("away_cover", probs["away_cover"], False, f"BB Alternate Spread: Away Cover ({-alt_spread:+.1f})"),
        ]
        if "over" in probs:
            specs.append(("over", probs["over"], False, f"BB Totals Over {total_line:.1f}"))
            specs.append(("under", probs["under"], False, f"BB Totals Under {total_line:.1f}"))
            # Team totals: P(home team scores more than home_total) via the
            # normal approximation (mean = lambda_home, std = 11).
            from scipy.stats import norm as _norm
            p_home_over = float(_norm.sf(home_total, loc=probs.get("expected_total", 0.0) / 2.0, scale=11.0))
            specs.append(("home_over_total", p_home_over, False,
                          f"BB Team Totals: Home Over {home_total:.1f}"))

        for key, prob, high_floor, label in specs:
            edge = 0.0
            if odds and key in odds and odds[key] > 1.0:
                implied = 1.0 / odds[key]
                edge = prob - implied
                if edge < self.min_edge:
                    continue
            candidates.append({
                "market": label,
                "key": key,
                "prob": float(prob),
                "high_floor": high_floor,
                "edge": edge,
                "odds": odds.get(key) if odds else None,
                "reason": f"{label} @ {prob:.1%}",
            })

        candidates.sort(key=lambda c: c["prob"], reverse=True)
        return candidates
