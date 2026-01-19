# %%
"""
Train an XGBoost model on NFL play-by-play data using an explicit
train/val/test split (80/10/10 overall) and early stopping.

Why this file exists:
- XGBoost + sklearn Pipeline + eval_set can break because eval_set bypasses
  the pipeline's preprocessor. This script uses "Option A":
  1) fit preprocessor on train only
  2) transform train/val/test
  3) fit XGBoost with eval_set on transformed data
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from xgboost import XGBRegressor

import matplotlib.pyplot as plt


# ------------------------------------------------------------------------------
# Local imports (repo)
# ------------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT))

from src.play_by_play.ml.features import build_features
from src.play_by_play.ml.labels import build_labels
from src.play_by_play.ml.util import convert_posteam_to_home_pred
from src.play_by_play.config.settings import settings


# ------------------------------------------------------------------------------
# Load data
# ------------------------------------------------------------------------------
# %%
data_dir = ROOT / "data" / "raw"

full_df = pd.DataFrame()
for i in range(1999, 2026):
    path = data_dir / f"play_by_play_{i}.parquet"
    df = pd.read_parquet(path)
    print(f"Year: {i}, Shape: {df.shape}")
    full_df = pd.concat([full_df, df], axis=0, ignore_index=True)

# ------------------------------------------------------------------------------
# Build features + labels
# ------------------------------------------------------------------------------
# %%
features = build_features(full_df)
labels = build_labels(full_df)

clean_df = pd.merge(features, labels, on=["game_id", "play_id"])

# ------------------------------------------------------------------------------
# Split X/y/keys
# ------------------------------------------------------------------------------
# %%
feature_cols = settings.schema.numeric_features + settings.schema.categorical_features
label_col = settings.schema.label_cols[0]
key_cols = settings.schema.key_cols

X = clean_df[feature_cols].copy()
y = clean_df[label_col].copy()
keys = clean_df[key_cols].copy()

# First split off test (10%)
X_tmp, X_test, y_tmp, y_test, keys_tmp, keys_test = train_test_split(
    X,
    y,
    keys,
    test_size=0.10,
    random_state=42,
)

# Then split remaining into train/val (val = 10% of total -> 0.10/0.90 = 0.111111...)
X_train, X_val, y_train, y_val, keys_train, keys_val = train_test_split(
    X_tmp,
    y_tmp,
    keys_tmp,
    test_size=0.1111111111,
    random_state=42,
)

print(f"Train: {X_train.shape}  Val: {X_val.shape}  Test: {X_test.shape}")

# ------------------------------------------------------------------------------
# Preprocessing
# ------------------------------------------------------------------------------
# %%
numeric_transformer = "passthrough"

categorical_transformer = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=True,  # sklearn >= 1.2
    # sparse=True,       # sklearn < 1.2
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, settings.schema.numeric_features),
        ("cat", categorical_transformer, settings.schema.categorical_features),
    ],
    remainder="drop",
)

# Fit preprocessor on TRAIN ONLY (avoid leakage), then transform all splits
preprocessor.fit(X_train)

X_train_t = preprocessor.transform(X_train)
X_val_t = preprocessor.transform(X_val)
X_test_t = preprocessor.transform(X_test)

# ------------------------------------------------------------------------------
# Model
# ------------------------------------------------------------------------------
# %%
model = XGBRegressor(
    n_estimators=800,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    objective="reg:squarederror",
    n_jobs=-1,
    random_state=42,
    early_stopping_rounds=50,
)

model.fit(
    X_train_t,
    y_train,
    eval_set=[(X_val_t, y_val)],
    verbose=False,
)

# ------------------------------------------------------------------------------
# Evaluate
# ------------------------------------------------------------------------------
# %%
y_pred = model.predict(X_test_t)

r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f"R^2 on test set: {r2:.3f}")
print(f"MSE on test set: {mse:.3f}")

# ------------------------------------------------------------------------------
# Analyze + plot
# ------------------------------------------------------------------------------
# %%
analyze = X_test.copy()
analyze["prediction"] = y_pred
analyze["prediction_home"] = convert_posteam_to_home_pred(analyze)
analyze["actual"] = y_test.to_numpy()  # keep alignment explicit
analyze = analyze.join(keys_test)

analyze["correct"] = np.select(
    [
        (analyze["prediction"] >= 0.5) & (analyze["actual"] >= 0.5),
        (analyze["prediction"] < 0.5) & (analyze["actual"] < 0.5),
    ],
    [1, 1],
    default=0,
)

# %%
# graph accuracy over different time buckets in the game
analyze["time_bucket"] = pd.cut(analyze["time_seconds_total"], bins=20)

analyze.groupby("time_bucket")["correct"].mean().plot(
    title="Prediction Accuracy over Time (Binned)"
)
plt.gca().invert_xaxis()
plt.show()

# %%
ten_games = (
    analyze[analyze["game_id"].str.contains("TEN")]["game_id"]
    .drop_duplicates()
    .sort_values()
)
ten_games

# %%
game = analyze[analyze["game_id"] == "2025_05_TEN_ARI"].sort_values(
    "time_seconds_total", ascending=False
)

plt.plot(game["time_seconds_total"], game["prediction_home"], label="Prediction")
plt.gca().invert_xaxis()
plt.show()
