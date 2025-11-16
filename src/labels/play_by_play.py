import numpy as np
import pandas as pd


def add_win_labels(df: pd.DataFrame) -> pd.DataFrame:
    # convert result (score) to win outcome
    df = df.copy()
    df["win"] = np.select(
        [df["result"] > 0,
        df["result"] == 0,
        df["result"] < 0],
        [1, 0.5, 0]
    )
    return df


def select_final_label_columns(df: pd.DataFrame) -> pd.DataFrame:
    key_cols = ["game_id", "play_id"]

    label_cols = [
        "win",
    ]
    return df[key_cols + label_cols]


def build_labels(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df
        .copy()
        .pipe(add_win_labels)
        .pipe(select_final_label_columns)
    )
