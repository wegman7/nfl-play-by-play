from pathlib import Path
import pandas as pd

from src.labels.play_by_play import build_labels


def run_build_labels():
    RAW_DATA_DIR = Path("data/raw")
    LABELS_DIR = Path("data/labels")

    cols = [
        "game_id",
        "play_id",
        "result"
    ]

    raw = pd.read_parquet(RAW_DATA_DIR / "play_by_play_2023.parquet")
    data = raw[cols].dropna()

    labels = build_labels(data)
    labels.to_parquet(LABELS_DIR / "play_by_play_2023.parquet")
