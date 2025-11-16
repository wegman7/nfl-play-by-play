# %%
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

ROOT = Path(__file__).resolve().parents[2]

# %%
key_cols = ["game_id", "play_id"]

numeric_features = [
    "qtr",
    "total_home_score",
    "total_away_score",
    "score_diff",
    "down",
    "ydstogo",
    "yardline_100",
    "posteam_timeouts_remaining",
    "defteam_timeouts_remaining",
    "time_seconds",
]

categorical_features = ["home_team", "posteam", "location"]

label_cols = ["win"]

# %%
FEATURES_PATH = ROOT / "data" / "features" / "play_by_play_2023.parquet"
LABELS_PATH = ROOT / "data" / "labels" / "play_by_play_2023.parquet"
features_raw = pd.read_parquet(FEATURES_PATH)
labels_raw = pd.read_parquet(LABELS_PATH)
full_dataset = features_raw.merge(
    labels_raw, 
    on=key_cols
)
features = full_dataset[numeric_features + categorical_features]
labels = full_dataset[label_cols]

# %%
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
X = features
y = labels
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Fit the model
clf.fit(X_train, y_train)

# Evaluate quickly
r2 = clf.score(X_test, y_test)
print(f"R^2 on test set: {r2:.3f}")
