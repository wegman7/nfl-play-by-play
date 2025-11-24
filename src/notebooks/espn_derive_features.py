# %%
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

import pandas as pd
from src.utils.espn_api_util import espn_game_to_df_with_timeouts

# %%
GAME_ID = "401772945"  # plug in any valid gameId

# %%
# Fetch game data with timeouts enriched
df = espn_game_to_df_with_timeouts(GAME_ID)

# %%
# Select columns to display
cols_to_show = [
    "sequenceNumber",
    "qtr",
    "clock",
    "text",
    "home_timeouts_remaining",
    "away_timeouts_remaining",
]
existing_cols = [c for c in cols_to_show if c in df.columns]

# %%
# Display the data
display_df = df[existing_cols]
print(display_df.head(60))

# %%
