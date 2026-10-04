"""Time-series validation.

Standard K-Fold leaks information in sequential sports data. This module
implements strict expanding-window splits: train on the past, validate on
the future, never the reverse.
"""
from __future__ import annotations

from typing import Iterator, List, Tuple

import pandas as pd


def expanding_window_splits(n: int, n_splits: int = 3,
                            min_train_frac: float = 0.5) -> Iterator[Tuple[range, range]]:
    """Yield (train_idx, val_idx) expanding-window splits.

    Each split trains on an expanding prefix and validates on the next
    contiguous block. All indices are chronological.
    """
    for i in range(1, n_splits + 1):
        train_end = int(n * min_train_frac) + int((n * (1 - min_train_frac)) * (i - 1) / n_splits)
        val_end = int(n * min_train_frac) + int((n * (1 - min_train_frac)) * i / n_splits)
        if train_end >= val_end or val_end > n:
            continue
        yield range(0, train_end), range(train_end, val_end)


def chronological_split(X: pd.DataFrame, y: pd.DataFrame, cutoff: float = 0.8):
    """Simple chronological train/test split by row fraction."""
    n = len(X)
    split = int(n * cutoff)
    return (X.iloc[:split], y.iloc[:split]), (X.iloc[split:], y.iloc[split:])


def evaluate_calibration(probs: pd.Series, labels: pd.Series, bins: int = 10) -> pd.DataFrame:
    """Calibration table: predicted vs actual frequency per bin."""
    df = pd.DataFrame({"prob": probs, "label": labels})
    df["bin"] = pd.qcut(df["prob"], bins, duplicates="drop")
    table = df.groupby("bin", observed=True).agg(
        mean_pred=("prob", "mean"),
        mean_actual=("label", "mean"),
        count=("label", "size"),
    ).reset_index()
    table["abs_error"] = (table["mean_pred"] - table["mean_actual"]).abs()
    return table
