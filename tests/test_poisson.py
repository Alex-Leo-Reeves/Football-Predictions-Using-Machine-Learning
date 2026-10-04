"""Tests for the Poisson / Dixon-Coles model."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.poisson import match_probabilities, score_matrix


def test_probabilities_sum_to_one():
    probs = match_probabilities(1.5, 1.1)
    total = probs["home_win"] + probs["draw"] + probs["away_win"]
    assert abs(total - 1.0) < 1e-6


def test_strong_home_favorite():
    probs = match_probabilities(2.5, 0.6)
    assert probs["home_win"] > probs["away_win"]
    assert probs["home_over_0_5"] > 0.9
    assert probs["over_1_5"] > 0.7


def test_low_scoring_match():
    probs = match_probabilities(0.7, 0.6)
    assert probs["under_1_5"] > 0.6
    assert probs["home_clean_sheet"] > 0.5


def test_matrix_shape():
    mat = score_matrix(1.5, 1.1, max_goals=6)
    assert mat.shape == (7, 7)
    assert abs(mat.sum() - 1.0) < 1e-6
