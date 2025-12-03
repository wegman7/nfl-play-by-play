# %%
from pathlib import Path
import sys
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT))

from src.play_by_play.ml.features import build_features
from src.play_by_play.ml.labels import build_labels

# %%
data_dir = ROOT / "data" / "raw"
full_df = pd.DataFrame()
for i in range(1999, 2026):
    path = data_dir / f"play_by_play_{i}.parquet"
    df = pd.read_parquet(path)
    print(f"Year: {i}, Shape: {df.shape}")
    full_df = pd.concat([full_df, df], axis=0, ignore_index=True)

# %%
full_df['win'] = np.select(
    [
        full_df['home_score'] > full_df['away_score'],   # home wins
        full_df['home_score'] < full_df['away_score'],   # home loses
    ],
    [
        1,   # win
        0,   # loss
    ],
    default=0.5  # tie
)

