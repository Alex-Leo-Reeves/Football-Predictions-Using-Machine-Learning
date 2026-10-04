"""DeepSeek fine-tuning (OpenAI-compatible API).

Fine-tuning runs on DeepSeek's infrastructure — no local disk or compute
cost, which fits the "no space on my system" constraint. This module:
  1. builds chat-format training examples from historical matches
  2. uploads a JSONL file
  3. creates + polls a fine-tuning job
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List

import requests

from src.config import get
from src.llm.prompts import SYSTEM_PROMPT


def build_training_examples(matches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert historical matches into chat-format fine-tuning examples.

    Each example: system prompt + user (feature summary) + assistant
    (the JSON prediction with the *actual* outcome as the target).
    """
    examples = []
    for m in matches:
        features = m.get("features", {})
        outcome = m.get("outcome", {})
        user_content = json.dumps({
            "fixture": m.get("fixture", ""),
            "features": {k: round(float(v), 4) for k, v in features.items()},
        }, default=str)
        assistant_content = json.dumps({
            "home_win_prob": float(outcome.get("home_win_prob", 0.0)),
            "draw_prob": float(outcome.get("draw_prob", 0.0)),
            "away_win_prob": float(outcome.get("away_win_prob", 0.0)),
            "home_over_0_5": float(outcome.get("home_over_0_5", 0.0)),
            "away_over_0_5": float(outcome.get("away_over_0_5", 0.0)),
            "over_1_5": float(outcome.get("over_1_5", 0.0)),
            "over_2_5": float(outcome.get("over_2_5", 0.0)),
            "btts_yes": float(outcome.get("btts_yes", 0.0)),
            "selected_market": outcome.get("selected_market", "NONE-SKIP"),
            "confidence": float(outcome.get("confidence", 0.0)),
            "reasoning": outcome.get("reasoning", ""),
            "decision": outcome.get("decision", "SKIP"),
        })
        examples.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content},
            ]
        })
    return examples


def write_jsonl(examples: List[Dict[str, Any]], path: str) -> None:
    """Write examples to a JSONL file for upload."""
    with open(path, "w", encoding="utf-8") as fh:
        for ex in examples:
            fh.write(json.dumps(ex) + "\n")


class DeepSeekFineTuner:
    """Submits and polls a DeepSeek fine-tuning job."""

    def __init__(self, api_key: str | None = None, base_url: str | None = None):
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY") or get("llm.api_key", None)
        # OpenAI-compatible fine-tuning on the SAME endpoint as inference
        # (dahl.global). Override with DEEPSEEK_FT_BASE_URL if your provider
        # exposes fine-tuning on a different host.
        self.base_url = base_url or os.getenv("DEEPSEEK_FT_BASE_URL") or get(
            "llm.fine_tune.base_url", "https://inference.dahl.global/v1")
        if not self.api_key:
            raise RuntimeError("DEEPSEEK_API_KEY not set")
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def upload_file(self, jsonl_path: str) -> str:
        """Upload a training file, returns file id."""
        url = f"{self.base_url}/files"
        with open(jsonl_path, "rb") as fh:
            resp = requests.post(
                url,
                headers={"Authorization": f"Bearer {self.api_key}"},
                files={"file": (os.path.basename(jsonl_path), fh, "application/jsonl")},
                data={"purpose": "fine-tune"},
                timeout=120,
            )
        resp.raise_for_status()
        return resp.json()["id"]

    def create_job(self, training_file: str, model: str | None = None,
                   suffix: str | None = None, n_epochs: int = 3) -> str:
        """Create a fine-tuning job, returns job id."""
        model = model or os.getenv("DEEPSEEK_FT_MODEL") or get("llm.fine_tune.base_model", "deepseek-chat")
        payload: Dict[str, Any] = {
            "model": model,
            "training_file": training_file,
            "hyperparameters": {"n_epochs": n_epochs},
        }
        if suffix:
            payload["suffix"] = suffix
        url = f"{self.base_url}/fine_tuning/jobs"
        resp = requests.post(url, headers=self.headers, json=payload, timeout=60)
        resp.raise_for_status()
        return resp.json()["id"]

    def poll_job(self, job_id: str, interval: int = 30, max_wait: int = 3600) -> Dict[str, Any]:
        """Poll until the job finishes (or times out)."""
        url = f"{self.base_url}/fine_tuning/jobs/{job_id}"
        waited = 0
        while waited < max_wait:
            resp = requests.get(url, headers=self.headers, timeout=30)
            resp.raise_for_status()
            job = resp.json()
            status = job.get("status")
            if status in ("succeeded", "failed", "cancelled"):
                return job
            time.sleep(interval)
            waited += interval
        return {"status": "timeout", "job_id": job_id}
