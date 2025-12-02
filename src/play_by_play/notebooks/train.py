# %%
from pathlib import Path
import sys
import tempfile

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

from mlflow.models import infer_signature
import mlflow
import mlflow.sklearn

ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT))

# --- MLflow setup (sqlite) ---
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("play_by_play_win_prob")

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

full_dataset = features_raw.merge(labels_raw, on=key_cols)
features = full_dataset[numeric_features + categorical_features]
labels = full_dataset[label_cols[0]]  # "win" as a Series

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

# %%
# MLflow tracking: fit, evaluate, log params/metrics/model + test set
with mlflow.start_run() as run:
    print("Run ID:", run.info.run_id)

    # log some basic params
    mlflow.log_params({
        "model_type": "RandomForestRegressor",
        "n_estimators": model.n_estimators,
        "random_state": model.random_state,
        "test_size": 0.2,
    })

    # Fit the model
    clf.fit(X_train, y_train)

    # Evaluate quickly
    r2 = clf.score(X_test, y_test)
    print(f"R^2 on test set: {r2:.3f}")

    # log metric
    mlflow.log_metric("r2", float(r2))

    # log X_test and y_test as artifacts
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        X_test_path = tmpdir_path / "X_test.parquet"
        y_test_path = tmpdir_path / "y_test.parquet"

        X_test.to_parquet(X_test_path)
        y_test.to_frame("win").to_parquet(y_test_path)

        # Store under "eval_data" in this run
        mlflow.log_artifact(str(X_test_path), artifact_path="eval_data")
        mlflow.log_artifact(str(y_test_path), artifact_path="eval_data")

    # log model with signature + example
    signature = infer_signature(X_train, clf.predict(X_train))

    mlflow.sklearn.log_model(
        clf,
        name="model",
        signature=signature,
        input_example=X_train.head(5),
    )
