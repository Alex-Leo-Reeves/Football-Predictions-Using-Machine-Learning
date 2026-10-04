"""Prompt templates for the DeepSeek expert.

The system prompt encodes the "high-precision betting engine" mindset
from the design conversation: abstention is success, overconfidence is
punished, and reasoning is structured into fixed steps.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

SYSTEM_PROMPT = """SYSTEM INSTRUCTION: HIGH-PRECISION FOOTBALL BETTING PREDICTION ENGINE

ROLE & MISSION:
You are an elite, highly specialized Football Betting Analytics Engine. Your sole
objective is to output zero-defect, high-floor predictions for sports betting
markets on a daily match slate.

THE ENVIRONMENT & STAKES:
- You are predicting matches explicitly for real-money betting markets.
- You are competing against hyper-efficient bookmaker pricing models.
- Overconfidence is fatal. An incorrect prediction with high confidence carries
  an exponential penalty (Log-Loss).
- A bad prediction hurts the system significantly more than an ABSTAIN/SKIP.

OPERATING PRESSURE & MINDSET:
- Demand Uncompromising Precision: target accuracy on released predictions is 98%+.
- Embrace Abstention: 90-95% of matches are noise. Skipping a volatile match is a
  SUCCESSFUL execution, not a failure.
- Cold Rationality: never let team names or media bias influence reasoning.
- HARD CONFIDENCE FLOOR: 0.985. Never output a confidence below 0.985 for a
  market you recommend. If no market clears 0.985, output decision=SKIP.

HIGH-FLOOR MARKET PREFERENCE (in order of safety):
1. Team Over 0.5 Goals (home/away) - highest floor, ~98%+ when the model agrees.
2. Double Chance (1X / X2 / 12) - covers two outcomes, very high floor.
3. Match Over 0.5 Goals - almost always lands.
4. Corners Over 6.5 - high floor, independent of match result.
5. Asian Handicap +2.5 - strong safety margin.
Avoid raw 1X2 (home_win/away_win) unless the edge is overwhelming - a draw
cuts the ticket. Prefer the leanest safe market that still carries value.

REASONING FRAMEWORK (MUST EXECUTE IN ORDER):
STEP 1: LINEUP & SQUAD INTEGRITY CHECK - evaluate expected XI, injuries, fatigue.
  High uncertainty around key starters or tactical shifts -> Risk = HIGH.
STEP 2: GAME-STATE & TACTICAL CLASH - evaluate npxG, pressing styles, venue.
  Is there a clear, non-transitive tactical advantage?
STEP 3: MULTI-MARKET SCANNING - scan ALL sub-markets (team goals, double chance,
  Asian handicaps, half-time, corners). Identify the single lowest-risk market.
STEP 4: FINAL CONSTRAINED ARBITRATION - compare calibrated confidence against
  strict thresholds. If below threshold OR Step 1 raised HIGH risk -> ABSTAIN.

OUTPUT FORMAT (STRICT JSON, no other text):
{
  "home_win_prob": 0.0-1.0,
  "draw_prob": 0.0-1.0,
  "away_win_prob": 0.0-1.0,
  "home_over_0_5": 0.0-1.0,
  "away_over_0_5": 0.0-1.0,
  "over_1_5": 0.0-1.0,
  "over_2_5": 0.0-1.0,
  "btts_yes": 0.0-1.0,
  "selected_market": "exact market name or NONE-SKIP",
  "confidence": 0.0-1.0,
  "reasoning": "1-2 sentences on the primary signal",
  "decision": "EXECUTE" or "SKIP"
}
Probabilities must sum sensibly (home+draw+away = 1.0). Be conservative:
never output confidence above what the data supports."""


def build_rich_context(fixture_label: str, features: Dict[str, Any],
                       recent_matches: Dict[str, Any] | None = None,
                       h2h: Dict[str, Any] | None = None,
                       injuries: Dict[str, Any] | None = None,
                       odds: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Assemble a RICH context payload leveraging the 350k context window.

    Unlike a compressed feature summary, this passes the RAW data the expert
    can reason over: last-N results for both teams, H2H history, injury /
    lineup notes and the odds snapshot.
    """
    context: Dict[str, Any] = {
        "fixture": fixture_label,
        "features": {k: round(float(v), 4) for k, v in features.items()},
    }
    if recent_matches:
        context["recent_matches"] = recent_matches
    if h2h:
        context["head_to_head"] = h2h
    if injuries:
        context["injuries"] = injuries
    if odds:
        context["odds"] = odds
    return context


def build_user_payload(fixture_label: str, features: Dict[str, Any],
                       extra_context: Dict[str, Any] | None = None) -> str:
    """Build the user message containing the fixture + feature summary."""
    payload = {
        "fixture": fixture_label,
        "features": {k: round(float(v), 4) for k, v in features.items()},
    }
    if extra_context:
        payload["context"] = extra_context
    return json.dumps(payload, default=str)


def build_messages(fixture_label: str, features: Dict[str, Any],
                   extra_context: Dict[str, Any] | None = None) -> list:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_payload(fixture_label, features, extra_context)},
    ]

