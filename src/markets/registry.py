"""Full market taxonomy.

The complete catalog of betting selections per match (from the design
conversation). The universal scanner iterates over these to find the
single lowest-risk entry per fixture.
"""
from __future__ import annotations

from typing import Dict, List

# ---------------------------------------------------------------------- #
# Market category definitions
# ---------------------------------------------------------------------- #
MATCH_OUTCOME_MARKETS: List[str] = [
    "1X2: Home Win", "1X2: Draw", "1X2: Away Win",
    "DC: 1X (Home/Draw)", "DC: X2 (Draw/Away)", "DC: 12 (Home/Away)",
    "DNB: Home Win (Draw Refund)", "DNB: Away Win (Draw Refund)",
    "Home to Win Either Half", "Away to Win Either Half",
    "Home to Win Both Halves", "Away to Win Both Halves",
    "Home Clean Sheet: Yes/No", "Away Clean Sheet: Yes/No",
    "Home Win to Nil", "Away Win to Nil",
]

GOAL_VOLUME_MARKETS: List[str] = [
    "OU Goals: Over/Under 0.5", "OU Goals: Over/Under 1.5",
    "OU Goals: Over/Under 2.5", "OU Goals: Over/Under 3.5",
    "OU Goals: Over/Under 4.5", "OU Goals: Over/Under 5.5",
    "Home Goals: Over/Under 0.5", "Home Goals: Over/Under 1.5", "Home Goals: Over/Under 2.5",
    "Away Goals: Over/Under 0.5", "Away Goals: Over/Under 1.5", "Away Goals: Over/Under 2.5",
    "BTTS: Yes (GG)", "BTTS: No (NG)",
    "BTTS + Over 2.5 Goals", "BTTS + Home/Away Win", "BTTS in Both Halves",
    "Exact Match Goals: 0, 1, 2, 3, 4, 5+",
    "Exact Team Goals: Home 0, 1, 2, 3+", "Exact Team Goals: Away 0, 1, 2, 3+",
    "Asian Total Goals: Over/Under 1.75", "Asian Total Goals: Over/Under 2.25",
    "Asian Total Goals: Over/Under 2.75",
    "Goal Range: 1-2 Goals", "Goal Range: 2-3 Goals", "Goal Range: 2-4 Goals",
]

HANDICAP_MARKETS: List[str] = [
    "AH Home/Away: 0.0 (DNB)",
    "AH Home/Away: -0.25 / +0.25", "AH Home/Away: -0.5 / +0.5",
    "AH Home/Away: -0.75 / +0.75", "AH Home/Away: -1.0 / +1.0",
    "AH Home/Away: -1.25 / +1.25", "AH Home/Away: -1.5 / +1.5",
    "AH Home/Away: -2.0 / +2.0", "AH Home/Away: -2.5 / +2.5",
    "EH Home (-1)", "EH Draw (-1)", "EH Away (+1)",
    "EH Home (-2)", "EH Draw (-2)", "EH Away (+2)",
]

TIME_AND_HALF_MARKETS: List[str] = [
    "1H 1X2: Home / Draw / Away", "1H Double Chance: 1X / X2 / 12",
    "1H Over/Under 0.5 Goals", "1H Over/Under 1.5 Goals",
    "2H 1X2: Home / Draw / Away", "2H Over/Under 0.5 Goals", "2H Over/Under 1.5 Goals",
    "HT/FT: 1/1", "HT/FT: X/1", "HT/FT: 2/1",
    "HT/FT: 1/X", "HT/FT: X/X", "HT/FT: 2/X",
    "HT/FT: 1/2", "HT/FT: X/2", "HT/FT: 2/2",
    "Highest Scoring Half: 1st Half / 2nd Half / Equal",
    "Goal Scored 1-15 Mins: Yes/No", "Goal Scored 1-30 Mins: Yes/No",
    "First 10 Mins Result: 1 / X / 2",
]

CORNER_AND_STATS_MARKETS: List[str] = [
    "Match Corners: Over/Under 6.5", "Match Corners: Over/Under 7.5",
    "Match Corners: Over/Under 8.5", "Match Corners: Over/Under 9.5",
    "Match Corners: Over/Under 10.5", "Match Corners: Over/Under 11.5",
    "Home Team Corners: Over/Under 3.5", "Home Team Corners: Over/Under 4.5",
    "Away Team Corners: Over/Under 3.5", "Away Team Corners: Over/Under 4.5",
    "Corner 1X2: Most Corners Home/Away", "Corner Asian Handicap (-1.5 / +1.5)",
    "First Corner: Home / Away", "Race to 3 / 5 / 7 Corners",
]

DISCIPLINE_MARKETS: List[str] = [
    "Total Match Cards: Over/Under 2.5", "Total Match Cards: Over/Under 3.5",
    "Total Match Cards: Over/Under 4.5", "Total Match Cards: Over/Under 5.5",
    "Home Cards: Over/Under 1.5", "Away Cards: Over/Under 1.5",
    "Red Card in Match: Yes/No", "Most Cards 1X2: Home / Draw / Away",
    "First Card Received: Home / Away",
    "Cards Points (Yellow=10, Red=25): Over/Under 20.5",
    "Cards Points (Yellow=10, Red=25): Over/Under 30.5",
]

PLAYER_PROP_MARKETS: List[str] = [
    "Player Total Shots: Over/Under 1.5, 2.5",
    "Player Shots on Target: Over 0.5, 1.5",
    "Anytime Goalscorer", "First Goalscorer", "Last Goalscorer", "Player to Score 2+ Goals",
    "Player Tackles: Over 1.5, 2.5",
    "Goalkeeper Saves: Over 2.5, 3.5, 4.5",
    "Player Carded: Yes/No",
    "Player to be Booked: Yes/No",
    "Player Passes: Over 45.5",
]

# ---------------------------------------------------------------------- #
# Basketball markets (no draws — moneyline, spread, totals)
# ---------------------------------------------------------------------- #
BASKETBALL_MARKETS: List[str] = [
    "BB Moneyline: Home Win", "BB Moneyline: Away Win",
    "BB 3-Way Result: Home / Regulation Tie / Away",
    "BB Double Result: Home/Home", "BB Double Result: Home/Away",
    "BB Double Result: Away/Home", "BB Double Result: Away/Away",
    "BB Spread: Home Cover", "BB Spread: Away Cover",
    "BB Alternate Spread: Home -12.5", "BB Alternate Spread: Away +12.5",
    "BB 1st Half Spread: Home Cover", "BB 2nd Half Spread: Home Cover",
    "BB 1st Quarter Spread: Home Cover",
    "BB Totals: Over 200.5", "BB Totals: Under 200.5",
    "BB Totals: Over 210.5", "BB Totals: Under 210.5",
    "BB Totals: Over 220.5", "BB Totals: Under 220.5",
    "BB Totals: Over 230.5", "BB Totals: Under 230.5",
    "BB Totals: Over 240.5", "BB Totals: Under 240.5",
    "BB Team Totals: Home Over 108.5", "BB Team Totals: Away Over 105.5",
    "BB 1st Half Totals: Over 112.5", "BB 1st Quarter Totals: Over 54.5",
    "BB Total Points: Odd / Even",
    "BB Winning Margin: 1-5", "BB Winning Margin: 6-10",
    "BB Winning Margin: 11-15", "BB Winning Margin: 16-20", "BB Winning Margin: 21+",
    "BB To Win All Quarters: Yes / No",
    "BB Highest Scoring Quarter: Q1 / Q2 / Q3 / Q4",
    "BB Race to 10 Points: Home / Away", "BB Race to 20 Points: Home / Away",
    "BB Player Points: Over 18.5", "BB Player Points: Over 24.5", "BB Player Points: Over 29.5",
    "BB Player Rebounds: Over 6.5", "BB Player Rebounds: Over 8.5",
    "BB Player Assists: Over 4.5", "BB Player Assists: Over 6.5",
    "BB Player 3PM: Over 1.5", "BB Player 3PM: Over 2.5",
    "BB Player PRA: Over 32.5", "BB Player PR: Over 25.5", "BB Player PA: Over 22.5",
    "BB Player Double-Double: Yes / No",
]

MARKET_CATEGORIES: Dict[str, List[str]] = {
    "match_outcome": MATCH_OUTCOME_MARKETS,
    "goal_volume": GOAL_VOLUME_MARKETS,
    "handicap": HANDICAP_MARKETS,
    "time_and_half": TIME_AND_HALF_MARKETS,
    "corners": CORNER_AND_STATS_MARKETS,
    "discipline": DISCIPLINE_MARKETS,
    "player_props": PLAYER_PROP_MARKETS,
    "basketball": BASKETBALL_MARKETS,
}


def all_markets() -> List[str]:
    """Flattened list of every market selection."""
    out: List[str] = []
    for markets in MARKET_CATEGORIES.values():
        out.extend(markets)
    return out


def market_count() -> int:
    return len(all_markets())
