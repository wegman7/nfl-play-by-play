# %%
import pandas as pd

# %%
raw = pd.read_parquet("play_by_play_2023.parquet")
train_cols = (
    pd.read_parquet("play_by_play_2023.parquet")
    [["game_id", "play_id", "qtr", "time", "score_differential", "home_team", "posteam", "down", "ydstogo", "yardline_100", "posteam_timeouts_remaining", "defteam_timeouts_remaining", "location"]]
).dropna()
train_cols
# %%
