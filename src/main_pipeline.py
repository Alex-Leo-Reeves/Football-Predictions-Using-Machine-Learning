"""Main daily prediction pipeline (headless, CI-friendly).

Flow:
  load config -> load data -> train engine -> build features ->
  master brain per fixture -> accumulator -> write output -> notify

Runs identically locally (lightweight backend) and on GitHub Actions
(heavy backend + optional DeepSeek expert).
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.config import get, load_config
from src.data.ingestion import DataLoader, filter_fixtures_by_time
from src.data.schemas import naive
from src.features.builder import FeatureBuilder
from src.markets.accumulator import AccumulatorBuilder
from src.models.master_brain import MasterBrain
from src.pipeline.training import train_basketball_engine, train_engine
from src.utils.logging_utils import get_logger
from src.utils.notify import notify_ticket

log = get_logger()


def _load_llm_expert():
    """Instantiate the DeepSeek expert only if a key is present."""
    if not os.getenv("DEEPSEEK_API_KEY") and not get("llm.api_key", None):
        log.info("No DEEPSEEK_API_KEY — running statistical-only (no LLM expert)")
        return None
    try:
        from src.llm.expert import LLMExpert
        return LLMExpert()
    except Exception as exc:  # pragma: no cover
        log.warning(f"LLM expert unavailable: {exc}")
        return None


def _build_llm_context(history, fixture, players, n_recent: int = 8) -> Dict[str, Any]:
    """Assemble RAW context blocks for the LLM expert (350k window).

    Returns recent form for both teams, H2H meetings, and injury/lineup
    availability — the raw data the expert reasons over.
    """
    # Normalize timezones so we never crash comparing naive vs aware datetimes.
    kickoff = naive(fixture.kickoff)
    prior = [m for m in history if naive(m.kickoff) < kickoff]
    home_id, away_id = fixture.home_team_id, fixture.away_team_id

    def team_recent(tid: str) -> List[Dict[str, Any]]:
        rows = []
        for m in sorted([x for x in prior if x.home_team_id == tid or x.away_team_id == tid],
                        key=lambda x: naive(x.kickoff))[-n_recent:]:
            if m.home_team_id == tid:
                gf, ga = m.home_goals, m.away_goals
                opp = m.away_team_id
            else:
                gf, ga = m.away_goals, m.home_goals
                opp = m.home_team_id
            rows.append({
                "date": m.kickoff.date().isoformat(),
                "opponent": opp,
                "for": gf, "against": ga,
                "xg_for": round(m.home_xg if m.home_team_id == tid else m.away_xg, 2),
                "xg_against": round(m.away_xg if m.home_team_id == tid else m.home_xg, 2),
            })
        return rows

    meetings = []
    for m in sorted([x for x in prior if {x.home_team_id, x.away_team_id} == {home_id, away_id}],
                    key=lambda x: naive(x.kickoff))[-6:]:
        meetings.append({
            "date": m.kickoff.date().isoformat(),
            "home": m.home_team_id, "away": m.away_team_id,
            "score": f"{m.home_goals}-{m.away_goals}",
        })

    injury_notes = {}
    if players:
        for tid, label in ((home_id, "home"), (away_id, "away")):
            squad = [p for p in players.values() if p.team_id == tid]
            injury_notes[label] = {
                "squad_size": len(squad),
                "kpai": round(fixture.home_kpai if label == "home" else fixture.away_kpai, 3),
            }

    return {
        "recent_matches": {"home": team_recent(home_id), "away": team_recent(away_id)},
        "h2h": meetings,
        "injuries": injury_notes,
    }


def run_daily_pipeline(config: Optional[Dict[str, Any]] = None,
                       use_llm: bool = True) -> Dict[str, Any]:
    """Execute the full daily prediction pipeline.

    Loads BOTH football and basketball fixtures, filters them to the
    remaining betting window (now -> end of day), trains a football engine
    and a basketball engine, evaluates every fixture, then builds a single
    accumulator targeting ``pipeline.target_odds`` (default 300).
    """
    config = config or load_config()
    loader = DataLoader()

    # ---- Load data (football + basketball) ----
    history = loader.load_history()
    fixtures = loader.load_today_fixtures()
    bb_history = loader.load_basketball_history()
    bb_fixtures = loader.load_today_basketball_fixtures()
    teams = loader.teams()
    players = loader.players()

    # ---- Time-window filter: only fixtures from now until end of day ----
    end_time = str(get("pipeline.kickoff_window.end", "23:30", config))
    fixtures = filter_fixtures_by_time(fixtures, end_time=end_time)
    bb_fixtures = filter_fixtures_by_time(bb_fixtures, end_time=end_time)

    log.info(f"Loaded {len(history)} football + {len(bb_history)} basketball historical matches, "
             f"{len(fixtures)} football + {len(bb_fixtures)} basketball today's fixtures")

    # ---- Train engines (heavy on CI, lightweight locally) ----
    backend = str(get("models.backend", "auto", config))
    engine = train_engine(history, backend=backend)
    bb_engine = None
    if bb_history:
        bb_engine = train_basketball_engine(bb_history, backend=backend)
    log.info(f"Football engine trained (backend={engine.backend}) metrics={engine.metrics}")
    if bb_engine is not None:
        log.info(f"Basketball engine trained (backend={bb_engine.backend}) metrics={bb_engine.metrics}")

    # ---- Features ----
    builder = FeatureBuilder(history, teams=teams, players=players)
    bb_builder = FeatureBuilder(bb_history, teams=loader.basketball_teams()) if bb_history else None

    # ---- Master brain (with optional DeepSeek expert) ----
    llm_expert = _load_llm_expert() if use_llm else None
    brain = MasterBrain(
        goal_worker=engine.goal_worker,
        corner_worker=engine.corner_worker,
        card_worker=engine.card_worker,
        basketball_worker=bb_engine.goal_worker if bb_engine is not None else None,
        anomaly_auditor=engine.anomaly_auditor,
        conformal=engine.conformal,
        llm_expert=llm_expert,
    )

    # ---- Evaluate every fixture (football + basketball) ----
    decisions: List[Dict[str, Any]] = []
    qualified: List[Dict[str, Any]] = []

    def _evaluate(fixture, feat_builder, hist, plyrs):
        """Evaluate one fixture, record the decision, and collect a qualified
        selection (with downgrade variants) when it EXECUTEs."""
        feats = feat_builder.fixture_features(fixture)
        llm_ctx = _build_llm_context(hist, fixture, plyrs)
        decision = brain.evaluate_fixture(
            fixture, feats,
            odds=fixture.odds or None,
            recent_matches=llm_ctx["recent_matches"],
            h2h=llm_ctx["h2h"],
            injuries=llm_ctx["injuries"],
        )
        decision["fixture_id"] = fixture.fixture_id
        decision["home_id"] = fixture.home_team_id
        decision["away_id"] = fixture.away_team_id
        decision["kickoff"] = fixture.kickoff.isoformat()
        decision["sport"] = fixture.sport
        decisions.append(decision)

        if decision["decision"] != "EXECUTE":
            return
        # Without live odds, fall back to model-estimated fair odds
        odds = decision.get("odds") or float(get("pipeline.default_odds", 1.15, config))
        # Build downgrade variants (leaner markets with lower odds) from the
        # ranked candidates so the accumulator can fine-tune toward 300.
        variants = []
        for v in decision.get("variants", []):
            v_odds = v.get("odds") or float(get("pipeline.default_odds", 1.15, config))
            variants.append({
                "market": v["market"],
                "key": v["key"],
                "confidence": round(v["prob"], 4),
                "odds": v_odds,
                "edge": v.get("edge", 0.0),
                "reason": v.get("reason", ""),
            })
        qualified.append({
            "fixture_id": fixture.fixture_id,
            "match": decision["fixture"],
            "home_id": fixture.home_team_id,
            "away_id": fixture.away_team_id,
            "sport": fixture.sport,
            "league": fixture.league,
            "country": fixture.country,
            "kickoff": fixture.kickoff.isoformat(),
            "market": decision["market"],
            "key": decision.get("key", ""),
            "confidence": decision["confidence"],
            "odds": odds,
            "edge": decision.get("edge", 0.0),
            "reason": decision.get("reason", ""),
            "llm_used": decision.get("llm_used", False),
            "variants": variants,
        })

    for fixture in fixtures:
        _evaluate(fixture, builder, history, players)
    if bb_builder is not None:
        for fixture in bb_fixtures:
            _evaluate(fixture, bb_builder, bb_history, {})

    # ---- Accumulator (over-select then optimize toward target odds) ----
    acc = AccumulatorBuilder()
    ticket = acc.build(qualified)

    # ---- Output ----
    output = {
        "date": datetime.now().isoformat(),
        "backend": engine.backend,
        "engine_metrics": engine.metrics,
        "basketball_engine_metrics": bb_engine.metrics if bb_engine is not None else None,
        "fixtures_evaluated": len(fixtures) + len(bb_fixtures),
        "football_fixtures_evaluated": len(fixtures),
        "basketball_fixtures_evaluated": len(bb_fixtures),
        "kickoff_window_end": end_time,
        "decisions": decisions,
        "qualified_selections": qualified,
        "ticket": ticket,
    }
    _write_output(output, config)
    notify_ticket({
        "date": output["date"],
        "total_legs": ticket["total_legs"],
        "combined_odds": ticket["combined_odds"],
        "estimated_probability": ticket["estimated_probability"],
        "selections": qualified,
    })
    return output


def _write_output(output: Dict[str, Any], config: Dict[str, Any]) -> None:
    out_dir = Path(get("output.dir", "output", config))
    out_dir.mkdir(parents=True, exist_ok=True)
    ticket_file = out_dir / get("output.ticket_file", "todays_ticket.json", config)
    report_file = out_dir / get("output.report_file", "daily_report.json", config)
    with open(ticket_file, "w", encoding="utf-8") as fh:
        json.dump(output["ticket"], fh, indent=2, default=str)
    with open(report_file, "w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2, default=str)
    log.info(f"Wrote {ticket_file} and {report_file}")


if __name__ == "__main__":
    result = run_daily_pipeline()
    print(json.dumps(result["ticket"], indent=2, default=str))
