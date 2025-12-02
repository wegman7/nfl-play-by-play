import pandas as pd
from play_by_play.utils.espn_api_util import espn_game_to_df_with_timeouts


def run_espn_derive_features():
    GAME_ID = "401772945"

    df = espn_game_to_df_with_timeouts(GAME_ID)

    cols_to_show = [
        "sequenceNumber",
        "qtr",
        "clock",
        "text",
        "home_timeouts_remaining",
        "away_timeouts_remaining",
    ]
    existing_cols = [c for c in cols_to_show if c in df.columns]

    display_df = df[existing_cols]
    print(display_df.head(60))
