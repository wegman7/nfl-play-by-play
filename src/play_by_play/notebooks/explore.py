# %%
from pathlib import Path
import sys
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT))

from src.play_by_play.ml.features import build_features
from src.play_by_play.ml.labels import build_labels
from src.play_by_play.config.settings import settings

# %%
data_dir = ROOT / "data" / "raw"
full_df = pd.DataFrame()
for i in range(1999, 2026):
    path = data_dir / f"play_by_play_{i}.parquet"
    df = pd.read_parquet(path)
    print(f"Year: {i}, Shape: {df.shape}")
    full_df = pd.concat([full_df, df], axis=0, ignore_index=True)

# %%
features = build_features(full_df)
labels = build_labels(full_df)
clean_df = pd.merge(features, labels, on=["game_id", "play_id"])

# %%
import pandas as pd
import matplotlib.pyplot as plt

def plot_win_rate_vs_score_diff(df, qtr=1, time_remaining=900, score_diff=10):
    df = df.copy()
    df = df[
        (df["score_diff"] >= -score_diff) & 
        (df["score_diff"] <= score_diff) &
        (df["time_seconds"] <= time_remaining) &
        (df["qtr"] == qtr)
    ]
    win_rate = df.groupby("score_diff")["win"].mean()

    # Plot
    plt.figure(figsize=(10, 5))
    win_rate.plot(kind="line")
    plt.axhline(0.5, color='gray', linestyle='--', linewidth=1)

    plt.title("Actual Win Rate vs Score Differential")
    plt.xlabel("Score Differential (Binned)")
    plt.ylabel("Win Probability")
    plt.grid(True)
    plt.show()

plot_win_rate_vs_score_diff(clean_df, time_remaining=900, score_diff=10, qtr=1)


# %%
import pandas as pd
import matplotlib.pyplot as plt

def plot_win_prob_over_time(df, score_diff, qtr, n_bins=20):
    """
    Plots win probability over time for a fixed score differential and quarter.

    df must include:
        - score_diff   (numeric)
        - qtr          (1-4)
        - time_seconds (seconds remaining in game or quarter)
        - win          (binary: 1 if home team eventually wins)
    """
    df = df.copy()

    # Filter down to plays in this quarter and this exact score differential
    df = df[(df["qtr"] == qtr) & (df["score_diff"] == score_diff)]
    print(f"Number of plays with score_diff={score_diff} in Q{qtr}: {len(df)}")

    if df.empty:
        print("No plays match this score differential and quarter.")
        return

    # Bin time into n_bins buckets
    df["time_bin"] = pd.cut(df["time_seconds"], bins=n_bins)

    # For each bin, compute mean win prob + mean time (bin midpoint)
    grouped = (
        df.groupby("time_bin")
          .agg(
              win_prob=("win", "mean"),
              mid_time=("time_seconds", "mean")
          )
          .sort_values("mid_time")
    )

    # Drop bins with no data (can happen if data is sparse)
    grouped = grouped.dropna(subset=["win_prob"])

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(grouped["mid_time"], grouped["win_prob"])

    plt.title(f"Win Probability Over Time\nScore Diff = {score_diff}, Q{qtr}")
    plt.xlabel("Time Remaining (seconds)")
    plt.ylabel("Win Probability")
    plt.grid(True)

    # Reverse x-axis so time flows left→right: start of quarter -> end
    plt.gca().invert_xaxis()

    plt.show()


# Example call
plot_win_prob_over_time(clean_df, score_diff=10, qtr=4)

# %%
num_cols = settings.schema.numeric_features
corr = clean_df.groupby('qtr')[num_cols + ['win']].corr()
corr['win']



# LOOK AT DIFFERENT ARCHITECTURES - DOES RANDOM FOREST WORK BETTER THAN NN?
# START WITH ONE FEATURE AND INCREASE
# HOW EXACTLY DOES RANDOM FOREST WORK??
# %%
