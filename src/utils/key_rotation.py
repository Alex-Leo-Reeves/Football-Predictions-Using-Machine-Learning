"""API key rotation pool.

Supports multiple API keys per provider (comma-separated env vars) with
per-key daily usage tracking persisted to disk. When one key exhausts its
budget, the pool rotates to the next — so 5 keys ≈ 5x the free-tier quota.

SECURITY: the persisted usage file stores only a SHA-256 hash of each key,
never the raw key.
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional


def parse_keys(*env_names: str) -> List[str]:
    """Collect keys from multiple env vars (comma-separated supported)."""
    keys: List[str] = []
    for name in env_names:
        raw = os.getenv(name, "")
        for part in raw.split(","):
            part = part.strip()
            if part and part not in keys:
                keys.append(part)
    return keys


def _hash_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()[:16]


class KeyPool:
    """Round-robin pool of keys with persistent daily usage tracking."""

    def __init__(self, keys: List[str], budget_per_key: int = 90,
                 state_file: str | Path | None = None):
        self.keys = keys
        self.budget_per_key = budget_per_key
        self.state_file = Path(state_file) if state_file else None
        self._index = 0
        # usage keyed by hashed key (never the raw key)
        self.usage: Dict[str, int] = {}
        self._load()

    # ------------------------------------------------------------------ #
    # Persistence (hashed keys only)
    # ------------------------------------------------------------------ #
    def _load(self) -> None:
        today = date.today().isoformat()
        if self.state_file and self.state_file.exists():
            try:
                data = json.loads(self.state_file.read_text())
                if data.get("date") == today:
                    self.usage = {k: int(v) for k, v in data.get("usage", {}).items()}
            except (json.JSONDecodeError, KeyError, ValueError):
                pass
        # ensure every key has an entry
        for key in self.keys:
            self.usage.setdefault(_hash_key(key), 0)

    def _save(self) -> None:
        if not self.state_file:
            return
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.state_file.write_text(json.dumps({
            "date": date.today().isoformat(),
            "usage": self.usage,
        }))

    # ------------------------------------------------------------------ #
    # API
    # ------------------------------------------------------------------ #
    @property
    def requests_left(self) -> int:
        return sum(max(0, self.budget_per_key - u) for u in self.usage.values())

    def acquire(self) -> Optional[str]:
        """Return the next key with remaining budget, or None if all spent."""
        if not self.keys:
            return None
        for _ in range(len(self.keys)):
            key = self.keys[self._index % len(self.keys)]
            self._index += 1
            if self.usage.get(_hash_key(key), 0) < self.budget_per_key:
                return key
        return None

    def record_use(self, key: str) -> None:
        self.usage[_hash_key(key)] = self.usage.get(_hash_key(key), 0) + 1
        self._save()

    def __len__(self) -> int:
        return len(self.keys)

