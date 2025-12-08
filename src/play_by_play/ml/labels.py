import numpy as np
import pandas as pd

from src.play_by_play.config.settings import settings


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
    key_cols = settings.schema.key_cols
    label_cols = settings.schema.label_cols
    return df[key_cols + label_cols]


def remove_na_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df[settings.schema.required_input_label_cols].dropna()


def build_labels(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df
        .copy()
        .pipe(remove_na_rows)
        .pipe(add_win_labels)
        .pipe(select_final_label_columns)
    )
