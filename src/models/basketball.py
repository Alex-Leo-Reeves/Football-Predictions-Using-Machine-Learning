"""Basketball probability model.

Basketball scoring is high-variance and continuous-ish, so we approximate
the point differential and total with normal distributions rather than
Poisson. From expected home/away points we derive:

  * moneyline probabilities (home_win / away_win — no draws in basketball)
  * point-spread probabilities (home covers / away covers)
  * over/under totals probabilities

The standard deviations are calibrated to typical NBA scoring variance
(~11-12 pts per team, ~16 pts on the differential, ~22 pts on the total).
"""
from __future__ import annotations

import math
from typing import Dict

from scipy.stats import norm


def basketball_probabilities(lambda_home: float, lambda_away: float,
                             spread: float = 0.0,
                             total_line: float | None = None) -> Dict[str, float]:
    """Derive basketball market probabilities from expected points.

    ``spread`` is the bookmaker line from the HOME team's perspective
    (e.g. -5.5 means home is favoured by 5.5 points). ``total_line`` is the
    over/under line (e.g. 220.5).
    """
    diff_mean = lambda_home - lambda_away
    diff_std = math.sqrt(11.0 ** 2 + 11.0 ** 2)  # ~15.6

    # P(home wins) = P(diff > 0)
    p_home = float(norm.sf(0.0, loc=diff_mean, scale=diff_std))
    p_away = 1.0 - p_home

    # Spread: home covers when diff > spread (i.e. diff - spread > 0)
    p_home_cover = float(norm.sf(spread, loc=diff_mean, scale=diff_std))
    p_away_cover = 1.0 - p_home_cover

    probs: Dict[str, float] = {
        "home_win": p_home,
        "away_win": p_away,
        "home_cover": p_home_cover,
        "away_cover": p_away_cover,
    }

    if total_line is not None:
        total_mean = lambda_home + lambda_away
        total_std = math.sqrt(11.0 ** 2 + 11.0 ** 2)  # ~15.6
        p_over = float(norm.sf(total_line, loc=total_mean, scale=total_std))
        probs["over"] = p_over
        probs["under"] = 1.0 - p_over
        probs["expected_total"] = total_mean

    probs["expected_total"] = lambda_home + lambda_away
    probs["expected_diff"] = diff_mean
    return probs
