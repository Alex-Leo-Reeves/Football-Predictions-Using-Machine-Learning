"""LLM expert — DeepSeek as a specialist in the hybrid ensemble.

The expert is invoked ONLY for fixtures that pass the initial statistical
gate (saving API cost), and its structured probabilities are blended with
the statistical workers by the master brain.

With V4 Flash's 350k context window, the expert can receive the RAW data
(recent form, H2H, injuries, odds) rather than just a compressed summary.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from src.config import get
from src.llm.client import DeepSeekClient
from src.llm.prompts import build_messages, build_rich_context


class LLMExpert:
    """Wraps DeepSeek into a predict_proba-like interface."""

    def __init__(self, client: DeepSeekClient | None = None, temperature: float | None = None,
                 max_tokens: int | None = None, rich_context: bool | None = None):
        self.client = client or DeepSeekClient()
        self.temperature = temperature if temperature is not None else float(get("llm.temperature", 0.0))
        self.max_tokens = max_tokens or int(get("llm.max_tokens", 8000))
        self.rich_context = rich_context if rich_context is not None else bool(get("llm.rich_context", True))

    def predict(self, fixture_label: str, features: Dict[str, Any],
                recent_matches: Dict[str, Any] | None = None,
                h2h: Dict[str, Any] | None = None,
                injuries: Dict[str, Any] | None = None,
                odds: Dict[str, Any] | None = None,
                extra_context: Dict[str, Any] | None = None) -> Dict[str, Any]:
        """Return structured probabilities + decision from the LLM."""
        if self.rich_context:
            context = build_rich_context(
                fixture_label, features,
                recent_matches=recent_matches,
                h2h=h2h,
                injuries=injuries,
                odds=odds,
            )
            if extra_context:
                context.update(extra_context)
            messages = build_messages(fixture_label, features, context)
        else:
            messages = build_messages(fixture_label, features, extra_context)

        raw = self.client.chat_json(
            messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return self._normalize(raw)

    @staticmethod
    def _normalize(raw: Dict[str, Any]) -> Dict[str, Any]:
        """Clamp probabilities into [0,1] and renormalize 1X2."""
        keys_1x2 = ["home_win_prob", "draw_prob", "away_win_prob"]
        total = sum(float(raw.get(k, 0.0)) for k in keys_1x2)
        if total <= 0:
            total = 1.0
        for k in keys_1x2:
            raw[k] = max(0.0, min(1.0, float(raw.get(k, 0.0)) / total))

        for k in ["home_over_0_5", "away_over_0_5", "over_1_5", "over_2_5", "btts_yes"]:
            raw[k] = max(0.0, min(1.0, float(raw.get(k, 0.0))))
        raw.setdefault("selected_market", "NONE-SKIP")
        raw.setdefault("confidence", 0.0)
        raw.setdefault("reasoning", "")
        raw.setdefault("decision", "SKIP")
        return raw
