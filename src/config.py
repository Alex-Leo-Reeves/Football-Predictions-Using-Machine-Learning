"""Configuration loader.

Loads ``config/config.yaml`` and overlays environment variables so the
same code runs locally (lightweight) and on GitHub Actions (heavy).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``override`` into ``base``."""
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: Path | str | None = None) -> Dict[str, Any]:
    """Load configuration from YAML, then overlay environment variables."""
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(path, "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh) or {}

    # Environment overlays (used by GitHub Actions / .env)
    env_overlay: Dict[str, Any] = {}
    if os.getenv("TARGET_ODDS"):
        env_overlay.setdefault("pipeline", {})["target_odds"] = float(os.getenv("TARGET_ODDS"))
    if os.getenv("MIN_CONFIDENCE"):
        env_overlay.setdefault("pipeline", {})["min_confidence"] = float(os.getenv("MIN_CONFIDENCE"))
    if os.getenv("ODDS_API_KEY"):
        env_overlay.setdefault("data", {})["odds_api_key"] = os.getenv("ODDS_API_KEY")
    if os.getenv("FOOTBALL_API_KEY"):
        env_overlay.setdefault("data", {})["football_api_key"] = os.getenv("FOOTBALL_API_KEY")

    return _deep_merge(config, env_overlay)


def get(key: str, default: Any = None, config: Dict[str, Any] | None = None) -> Any:
    """Dot-notation getter, e.g. get('models.conformal.alpha')."""
    cfg = config if config is not None else load_config()
    node: Any = cfg
    for part in key.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node
