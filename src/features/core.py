import pandas as pd

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

def select_final_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    key_cols = ["game_id", "play_id"]

    feature_cols = [
        "qtr",
        "score_diff",
        "total_home_score",
        "total_away_score",
        "down",
        "ydstogo",
        "yardline_100",
        "posteam_timeouts_remaining",
        "defteam_timeouts_remaining",
        "time_seconds",
        "home_team",
        "posteam",
        "location",
    ]
    return df[key_cols + feature_cols]

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df
        .copy()
        .pipe(add_time_features)
        .pipe(add_score_features)
        .pipe(select_final_feature_columns)
    )
