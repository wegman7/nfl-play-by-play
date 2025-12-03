import pandas as pd

from play_by_play.config.settings import settings

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    # your mm:ss → seconds logic, etc.
    df = df.copy()
    print(df.head())
    def mmss_to_seconds(t: str) -> int:
        m, s = t.split(":")
        return int(m) * 60 + int(s)
    df["time_seconds"] = df["time"].astype(str).apply(mmss_to_seconds)
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


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df
        .copy()
        .pipe(add_time_features)
        .pipe(add_score_features)
        .pipe(convert_posteam_to_is_home)
        .pipe(select_final_feature_columns)
    )
