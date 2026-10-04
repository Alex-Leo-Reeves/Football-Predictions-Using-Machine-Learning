"""Model backend abstraction.

Locally (no disk space for heavy deps) we fall back to scikit-learn's
HistGradientBoosting. On GitHub Actions the same code transparently uses
XGBoost or LightGBM for heavier training. All backends expose a
scikit-learn-compatible API.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from src.config import get


def resolve_backend(backend: str = "auto") -> str:
    """Return the concrete backend name to use."""
    if backend != "auto":
        return backend
    for candidate in ("xgboost", "lightgbm", "hist"):
        try:
            _import_backend(candidate)
            return candidate
        except ImportError:
            continue
    return "hist"


def _import_backend(name: str):
    if name == "xgboost":
        import xgboost  # noqa: F401
    elif name == "lightgbm":
        import lightgbm  # noqa: F401
    else:
        from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: F401


def make_regressor(backend: str = "auto", **kwargs: Any):
    """Create a gradient-boosted regressor on the resolved backend."""
    backend = resolve_backend(backend)
    params = dict(get("models.worker", {}))
    params.update(kwargs)

    if backend == "xgboost":
        import xgboost as xgb
        return xgb.XGBRegressor(
            n_estimators=params.get("n_estimators", 300),
            learning_rate=params.get("learning_rate", 0.05),
            max_depth=params.get("max_depth", 4),
            subsample=params.get("subsample", 0.9),
            colsample_bytree=params.get("colsample_bytree", 0.9),
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1,
        )
    if backend == "lightgbm":
        import lightgbm as lgb
        return lgb.LGBMRegressor(
            n_estimators=params.get("n_estimators", 300),
            learning_rate=params.get("learning_rate", 0.05),
            max_depth=params.get("max_depth", 4),
            subsample=params.get("subsample", 0.9),
            colsample_bytree=params.get("colsample_bytree", 0.9),
            random_state=42,
            n_jobs=-1,
        )
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(
        max_iter=params.get("n_estimators", 300),
        learning_rate=params.get("learning_rate", 0.05),
        max_depth=params.get("max_depth", 4),
        random_state=42,
    )


def make_classifier(backend: str = "auto", **kwargs: Any):
    """Create a gradient-boosted classifier on the resolved backend."""
    backend = resolve_backend(backend)
    params = dict(get("models.worker", {}))
    params.update(kwargs)

    if backend == "xgboost":
        import xgboost as xgb
        return xgb.XGBClassifier(
            n_estimators=params.get("n_estimators", 300),
            learning_rate=params.get("learning_rate", 0.05),
            max_depth=params.get("max_depth", 4),
            subsample=params.get("subsample", 0.9),
            colsample_bytree=params.get("colsample_bytree", 0.9),
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        )
    if backend == "lightgbm":
        import lightgbm as lgb
        return lgb.LGBMClassifier(
            n_estimators=params.get("n_estimators", 300),
            learning_rate=params.get("learning_rate", 0.05),
            max_depth=params.get("max_depth", 4),
            subsample=params.get("subsample", 0.9),
            colsample_bytree=params.get("colsample_bytree", 0.9),
            random_state=42,
            n_jobs=-1,
        )
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(
        max_iter=params.get("n_estimators", 300),
        learning_rate=params.get("learning_rate", 0.05),
        max_depth=params.get("max_depth", 4),
        random_state=42,
    )
