import pandas as pd

from src.labels.play_by_play import build_labels
from config.settings import settings


def run_build_labels():
    cols = [
        "game_id",
        "play_id",
        "result"
    ]

    raw = pd.read_parquet(settings.paths.raw_play_by_play_2023)
    data = raw[cols].dropna()

    labels = build_labels(data)
    settings.paths.labels_dir.mkdir(parents=True, exist_ok=True)
    labels.to_parquet(settings.paths.labels_2023)
