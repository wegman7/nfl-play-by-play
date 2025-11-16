# %%
import os

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
# %%

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

data_path = os.path.join(ROOT, "data", "raw", "play_by_play_2023.parquet")

raw = pd.read_parquet(data_path)

# Feature columns you mentioned
feature_cols = [
    "game_id",
    "play_id",
    "qtr",
    "time",
    "total_home_score",
    "total_away_score",
    "home_team",
    "posteam",
    "down",
    "ydstogo",
    "yardline_100",
    "posteam_timeouts_remaining",
    "defteam_timeouts_remaining",
    "location",
]

data = raw[feature_cols + ["result"]].dropna()
# %% convert result (score) to win outcome
data["win"] = np.select(
    [data["result"] > 0,
     data["result"] == 0,
     data["result"] < 0],
    [1, 0.5, 0]
)
# %%

# Convert "MM:SS" in 'time' to seconds remaining in the quarter
def mmss_to_seconds(t):
    m, s = t.split(":")
    return int(m) * 60 + int(s)

data["time_seconds"] = data["time"].astype(str).apply(mmss_to_seconds)

# Now define X, y
X = data.drop(columns=["win", "result", "game_id", "play_id", "time"])
y = data["win"]
# %%

# Identify column types
numeric_features = [
    "qtr",
    "total_home_score",
    "total_away_score",
    "down",
    "ydstogo",
    "yardline_100",
    "posteam_timeouts_remaining",
    "defteam_timeouts_remaining",
    "time_seconds",
]

categorical_features = ["home_team", "posteam", "location"]

# Preprocessor: OneHot encode categoricals, pass through numerics
numeric_transformer = "passthrough"

categorical_transformer = OneHotEncoder(
    handle_unknown="ignore",  # avoids crashing on unseen teams/locations
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

# Define the model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1,
)

# Full pipeline: preprocess -> model
clf = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model),
    ]
)
# %%

# train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Fit the model
clf.fit(X_train, y_train)

# Evaluate quickly
r2 = clf.score(X_test, y_test)
print(f"R^2 on test set: {r2:.3f}")
# %% inference

y_pred = clf.predict(X_test)
pred_series = pd.Series(y_pred, index=X_test.index, name="win_pred")

df_with_preds = X_test.join(pred_series, how="left")
df_with_preds["win_true"] = y_test