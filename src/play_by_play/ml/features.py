import numpy as np
import pandas as pd

from src.play_by_play.config.settings import settings

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    # your mm:ss → seconds logic, etc.
    df = df.copy()
    def mmss_to_seconds(t: str) -> int:
        m, s = t.split(":")
        return int(m) * 60 + int(s)
    df["time_seconds"] = df["time"].astype(str).apply(mmss_to_seconds)
    return df


def add_seconds_total_feature(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # handle overtime
    remaining_quarters = np.maximum(4 - df["qtr"], 0)
    df["time_seconds_total"] = (
        remaining_quarters * 15 * 60 +  # full quarters
        df["time_seconds"]            # seconds in current quarter
    )
    return df


def add_score_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["score_diff"] = df["total_home_score"] - df["total_away_score"]
    return df


def convert_posteam_to_is_home(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['posteam_is_home'] = df['posteam'] == df['home_team']
    return df


def select_final_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    key_cols = settings.schema.key_cols
    feature_cols = settings.schema.all_feature_cols
    return df[key_cols + feature_cols]


def remove_na_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df[settings.schema.required_input_feature_cols].dropna()


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df
        .copy()
        .pipe(remove_na_rows)
        .pipe(add_time_features)
        .pipe(add_seconds_total_feature)
        .pipe(add_score_features)
        .pipe(convert_posteam_to_is_home)
        .pipe(select_final_feature_columns)
    )
