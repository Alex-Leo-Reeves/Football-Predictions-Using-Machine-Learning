#!/usr/bin/env python3
"""Prepare + submit a DeepSeek fine-tuning job (runs on DeepSeek servers).

Usage:
  python scripts/fine_tune_deepseek.py --examples 200 --epochs 3

Requires DEEPSEEK_API_KEY. Training data is built from the loaded history
(synthetic by default; swap in real data via the DataLoader).
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.ingestion import DataLoader
from src.features.builder import FeatureBuilder
from src.llm.fine_tune import DeepSeekFineTuner, build_training_examples, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--examples", type=int, default=200, help="max training examples")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--suffix", default="football-v1")
    parser.add_argument("--submit", action="store_true", help="actually submit the job")
    args = parser.parse_args()

    loader = DataLoader()
    history = loader.load_history()
    builder = FeatureBuilder(history)

    # Build feature summaries + outcome targets for the most recent matches
    matches = []
    for match in history[-args.examples:]:
        from src.data.schemas import Fixture
        fixture = Fixture(
            fixture_id=match.match_id, kickoff=match.kickoff,
            home_team_id=match.home_team_id, away_team_id=match.away_team_id,
            home_kpai=match.home_kpai, away_kpai=match.away_kpai,
        )
        feats = builder.fixture_features(fixture)
        matches.append({
            "fixture": f"{match.home_team_id} vs {match.away_team_id}",
            "features": feats,
            "outcome": {
                "home_win_prob": 1.0 if match.outcome == "H" else 0.0,
                "draw_prob": 1.0 if match.outcome == "D" else 0.0,
                "away_win_prob": 1.0 if match.outcome == "A" else 0.0,
                "home_over_0_5": 1.0 if match.home_goals > 0 else 0.0,
                "away_over_0_5": 1.0 if match.away_goals > 0 else 0.0,
                "over_1_5": 1.0 if match.total_goals > 1 else 0.0,
                "over_2_5": 1.0 if match.total_goals > 2 else 0.0,
                "btts_yes": 1.0 if (match.home_goals > 0 and match.away_goals > 0) else 0.0,
                "selected_market": "NONE-SKIP",
                "confidence": 0.0,
                "reasoning": "",
                "decision": "SKIP",
            },
        })

    examples = build_training_examples(matches)
    print(f"Built {len(examples)} training examples")

    if not args.submit:
        print("Dry run — pass --submit to upload and start the job.")
        return

    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tmp:
        write_jsonl(examples, tmp.name)
        tuner = DeepSeekFineTuner()
        file_id = tuner.upload_file(tmp.name)
        print(f"Uploaded training file: {file_id}")
        job_id = tuner.create_job(file_id, suffix=args.suffix, n_epochs=args.epochs)
        print(f"Fine-tuning job created: {job_id}")
        print("Polling (this can take a while)...")
        status = tuner.poll_job(job_id, interval=30, max_wait=3600)
        print(json.dumps(status, indent=2))
        # The fine-tuned model id is what the pipeline should use for
        # inference — set it as DEEPSEEK_MODEL to make the fine-tuned
        # brain drive predictions (one big brain, not a separate job).
        ft_model = status.get("fine_tuned_model") or status.get("model")
        if status.get("status") == "succeeded" and ft_model:
            print("\n✅ Fine-tuning succeeded!")
            print(f"Fine-tuned model id: {ft_model}")
            print("To use it in the pipeline, set the GitHub secret:")
            print(f"  DEEPSEEK_MODEL = {ft_model}")


if __name__ == "__main__":
    main()
