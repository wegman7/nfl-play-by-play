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

from play_by_play.config.settings import settings

# --- MLflow setup (sqlite) ---
mlflow.set_tracking_uri(settings.mlflow.tracking_uri)

# Create or get experiment with artifact location at project root
experiment = mlflow.get_experiment_by_name(settings.mlflow.experiment_name)
if experiment is None:
    mlflow.create_experiment(
        settings.mlflow.experiment_name,
        artifact_location=settings.mlflow.artifact_location
    )
mlflow.set_experiment(settings.mlflow.experiment_name)

# %%
key_cols = settings.schema.key_cols
numeric_features = settings.schema.numeric_features
categorical_features = settings.schema.categorical_features
label_cols = settings.schema.label_cols

# %%
FEATURES_PATH = settings.paths.features_2023
LABELS_PATH = settings.paths.labels_2023
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
    n_estimators=settings.training.model_config.n_estimators,
    random_state=settings.training.model_config.random_state,
    n_jobs=settings.training.model_config.n_jobs,
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
    X, y,
    test_size=settings.training.test_size,
    random_state=settings.training.random_state
)

# %%
# MLflow tracking: fit, evaluate, log params/metrics/model + test set
with mlflow.start_run() as run:
    print("Run ID:", run.info.run_id)

    # log some basic params
    mlflow.log_params({
        "model_type": "RandomForestRegressor",
        "n_estimators": settings.training.model_config.n_estimators,
        "random_state": settings.training.model_config.random_state,
        "test_size": settings.training.test_size,
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

# %%
