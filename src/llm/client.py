"""DeepSeek API client (OpenAI-compatible).

Used as a *specialist expert* in the hybrid architecture — NOT for every
prediction. The client is dependency-light (plain `requests`) and the
model name is configurable via env ``DEEPSEEK_MODEL``.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import requests

from src.config import get


class DeepSeekClient:
    def __init__(self, api_key: str | None = None, model: str | None = None,
                 base_url: str | None = None, timeout: int = 30):
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY") or get("llm.api_key", None)
        self.model = model or os.getenv("DEEPSEEK_MODEL") or get(
            "llm.model", "deepseek-ai/DeepSeek-V4-Flash-0731")
        self.base_url = base_url or os.getenv("DEEPSEEK_BASE_URL") or get(
            "llm.base_url", "https://inference.dahl.global/v1")
        self.timeout = timeout
        if not self.api_key:
            raise RuntimeError(
                "DEEPSEEK_API_KEY not set. Add it to your environment or .env file."
            )

    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.0,
             max_tokens: int = 800, json_mode: bool = True) -> str:
        """Single chat completion call. Returns the assistant text."""
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        resp = requests.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=payload,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]

    def chat_json(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """Chat completion parsed as JSON."""
        text = self.chat(messages, **kwargs)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # tolerate markdown fences
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(text[start:end])
            raise
