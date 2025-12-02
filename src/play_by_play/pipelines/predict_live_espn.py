"""
ESPN API-based live win probability predictor.

Polls ESPN API every X seconds, applies feature engineering,
and generates win probability predictions using the best MLflow model.
"""

import time
import pandas as pd
from pathlib import Path
from datetime import datetime
import argparse

from play_by_play.utils.espn_api_util import espn_game_to_df_with_timeouts
from play_by_play.ml.features import build_features
from play_by_play.utils.model_util import load_best_model_from_experiment
from play_by_play.config.settings import settings


def predict_from_espn(
    game_id: str,
    model,
    output_dir: Path,
    poll_interval: int = 10,
    max_iterations: int = None,
):
    """
    Poll ESPN API for live game data and generate win probability predictions.

    Args:
        game_id: ESPN game ID to track
        model: Trained sklearn model for predictions
        output_dir: Directory to save predictions
        poll_interval: Seconds between ESPN API polls
        max_iterations: Maximum number of polling iterations (None = infinite)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    iteration = 0
    seen_play_ids = set()

    print(f"Starting ESPN polling for game {game_id}")
    print(f"Polling interval: {poll_interval} seconds")
    print(f"Output directory: {output_dir}")
    print("-" * 60)

    try:
        while True:
            iteration += 1
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            print(f"[{timestamp}] Iteration {iteration}: Fetching ESPN data...")

            try:
                raw_df = espn_game_to_df_with_timeouts(game_id)

                if raw_df.empty:
                    print(f"  No plays found. Waiting {poll_interval}s...")
                    time.sleep(poll_interval)
                    continue

                features_df = build_features(raw_df)

                if "play_id" in features_df.columns:
                    new_plays = features_df[~features_df["play_id"].isin(seen_play_ids)]

                    if new_plays.empty:
                        print(f"  No new plays. Total plays tracked: {len(seen_play_ids)}")
                        time.sleep(poll_interval)
                        continue

                    seen_play_ids.update(new_plays["play_id"].tolist())
                else:
                    new_plays = features_df

                valid_plays = new_plays.dropna(subset=settings.schema.all_feature_cols)
                num_invalid = len(new_plays) - len(valid_plays)

                if num_invalid > 0:
                    print(f"  Skipping {num_invalid} plays with missing features")

                if valid_plays.empty:
                    print(f"  No valid plays to predict")
                    time.sleep(poll_interval)
                    continue

                X = valid_plays[settings.schema.all_feature_cols]
                predictions = model.predict(X)
                valid_plays = valid_plays.copy()
                valid_plays["win_prob"] = predictions
                valid_plays["prediction_timestamp"] = timestamp

                output_file = output_dir / "espn_predictions.parquet"

                if output_file.exists():
                    existing_df = pd.read_parquet(output_file)
                    combined_df = pd.concat([existing_df, valid_plays], ignore_index=True)
                    combined_df.to_parquet(output_file, index=False)
                else:
                    valid_plays.to_parquet(output_file, index=False)

                print(f"  ✓ Predicted {len(valid_plays)} new plays")
                print(f"  Latest play: Q{valid_plays.iloc[-1]['qtr']} {valid_plays.iloc[-1]['time_seconds']}")
                print(f"  Win prob: {predictions[-1]:.3f}")
                print(f"  Total plays tracked: {len(seen_play_ids)}")

            except Exception as e:
                print(f"  Error during iteration {iteration}: {e}")
                print(f"  Continuing to next iteration...")

            if max_iterations and iteration >= max_iterations:
                print(f"\nReached max iterations ({max_iterations}). Stopping.")
                break

            time.sleep(poll_interval)

    except KeyboardInterrupt:
        print("\n\nStopped by user (Ctrl+C)")
        print(f"Total iterations: {iteration}")
        print(f"Total plays tracked: {len(seen_play_ids)}")


def run_predict_live_espn():
    parser = argparse.ArgumentParser(
        description="Poll ESPN API and generate live win probability predictions"
    )
    parser.add_argument(
        "--game-id",
        type=str,
        default=settings.espn.default_game_id,
        help=f"ESPN game ID (default: '{settings.espn.default_game_id}')"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=settings.espn.poll_interval_seconds,
        help=f"Polling interval in seconds (default: {settings.espn.poll_interval_seconds})"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=f"Output directory for predictions (default: {settings.paths.live_espn_dir})"
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        help="Maximum number of polling iterations (default: unlimited)"
    )

    args = parser.parse_args()

    print("Loading best model from MLflow...")
    model, model_uri, run_id = load_best_model_from_experiment(
        experiment_name=settings.mlflow.experiment_name,
        tracking_uri=settings.mlflow.tracking_uri,
        metric=settings.mlflow.metric,
        higher_is_better=settings.mlflow.metric_higher_is_better,
    )
    print(f"Loaded model from {model_uri} (run_id={run_id})")
    print()

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = settings.paths.live_espn_dir

    predict_from_espn(
        game_id=args.game_id,
        model=model,
        output_dir=output_dir,
        poll_interval=args.interval,
        max_iterations=args.max_iterations,
    )
