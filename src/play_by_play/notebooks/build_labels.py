# %%
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT))

import pandas as pd

from play_by_play.ml.labels import build_labels

# %%
RAW_DATA_DIR = ROOT / "data" / "raw"
LABELS_DIR = ROOT / "data" / "labels"
# %%
cols = [
    "game_id",
    "play_id",
    "result"
]

raw = pd.read_parquet(RAW_DATA_DIR / "play_by_play_2023.parquet")
data = raw[cols].dropna()

# %%
labels = build_labels(data)
labels.to_parquet(LABELS_DIR / "play_by_play_2023.parquet")
