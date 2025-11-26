from pathlib import Path
import pandas as pd

from src.features.play_by_play import build_features


def run_build_features():
    RAW_DATA_DIR = Path("data/raw")
    FEATURES_DIR = Path("data/features")

    cols = [
        "game_id",
        "play_id",
        "qtr",
        "time",
        "total_home_score",
        "total_away_score",
        "home_team",
        "posteam",
        "down",
        "ydstogo",
        "yardline_100",
        "posteam_timeouts_remaining",
        "defteam_timeouts_remaining",
        "location",
    ]

    raw = pd.read_parquet(RAW_DATA_DIR / "play_by_play_2023.parquet")
    data = raw[cols].dropna()

    features = build_features(data)
    features.to_parquet(FEATURES_DIR / "play_by_play_2023.parquet")
