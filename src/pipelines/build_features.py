import pandas as pd

from src.features.play_by_play import build_features
from config.settings import settings


def run_build_features():
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

    raw = pd.read_parquet(settings.paths.raw_play_by_play_2023)
    data = raw[cols].dropna()

    features = build_features(data)
    settings.paths.features_dir.mkdir(parents=True, exist_ok=True)
    features.to_parquet(settings.paths.features_2023)
