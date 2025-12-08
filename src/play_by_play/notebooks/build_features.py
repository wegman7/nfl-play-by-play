# %%
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT))

import pandas as pd

from play_by_play.ml.features import build_features

# %%
RAW_DATA_DIR = ROOT / "data" / "raw"
FEATURES_DIR = ROOT / "data" / "features"

# %%
raw = pd.read_parquet(RAW_DATA_DIR / "play_by_play_2023.parquet")

# %%
features = build_features(raw)
features.to_parquet(FEATURES_DIR / "play_by_play_2023.parquet")
