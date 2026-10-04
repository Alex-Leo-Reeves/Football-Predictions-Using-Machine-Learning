#!/usr/bin/env python3
"""Test the DeepSeek connection: list available models + send a tiny probe.

Usage:
  DEEPSEEK_API_KEY=sk-... python3 scripts/test_deepseek.py

This confirms your key works and shows the exact model ids available on
your account (so you can set DEEPSEEK_MODEL correctly for "V4 Flash").
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import requests


def main() -> None:
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("ERROR: DEEPSEEK_API_KEY not set.")
        print("  export DEEPSEEK_API_KEY=sk-...   (or put it in .env)")
        sys.exit(1)

    base = os.getenv("DEEPSEEK_BASE_URL", "https://inference.dahl.global/v1")
    headers = {"Authorization": f"Bearer {api_key}"}

    # 1. List available models (may not be supported on all inference proxies)
    print("== Available models ==")
    try:
        resp = requests.get(f"{base}/models", headers=headers, timeout=15)
        resp.raise_for_status()
        models = resp.json().get("data", [])
        for m in models:
            print(f"  - {m.get('id')}")
        if not models:
            print("  (no models returned)")
    except requests.RequestException as exc:
        print(f"  (model list not available on this endpoint: {exc})")

    # 2. Tiny probe chat
    print("\n== Probe chat ==")
    try:
        resp = requests.post(
            f"{base}/chat/completions",
            headers={**headers, "Content-Type": "application/json"},
            json={
                "model": os.getenv("DEEPSEEK_MODEL", "deepseek-ai/DeepSeek-V4-Flash-0731"),
                "messages": [{"role": "user", "content": "Reply with exactly: OK"}],
                "max_tokens": 5,
            },
            timeout=30,
        )
        resp.raise_for_status()
        reply = resp.json()["choices"][0]["message"]["content"]
        print(f"  Model replied: {reply!r}")
        print("  ✅ DeepSeek key works.")
    except requests.RequestException as exc:
        print(f"  ❌ Probe failed: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
