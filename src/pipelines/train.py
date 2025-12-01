from pathlib import Path
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

from config.settings import settings


def run_train():
    mlflow.set_tracking_uri(settings.mlflow.tracking_uri)
    mlflow.set_experiment(settings.mlflow.experiment_name)

    features_raw = pd.read_parquet(settings.paths.features_2023)
    labels_raw = pd.read_parquet(settings.paths.labels_2023)

    full_dataset = features_raw.merge(labels_raw, on=settings.schema.key_cols)
    features = full_dataset[settings.schema.all_feature_cols]
    labels = full_dataset[settings.schema.label_cols[0]]

    numeric_transformer = "passthrough"

    categorical_transformer = OneHotEncoder(
        handle_unknown="ignore",
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, settings.schema.numeric_features),
            ("cat", categorical_transformer, settings.schema.categorical_features),
        ]
    )

    model = RandomForestRegressor(**settings.training.model_config.__dict__)

    clf = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    X = features
    y = labels

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=settings.training.test_size, random_state=settings.training.random_state
    )

    with mlflow.start_run() as run:
        print("Run ID:", run.info.run_id)

        mlflow.log_params({
            "model_type": "RandomForestRegressor",
            "n_estimators": model.n_estimators,
            "random_state": model.random_state,
            "test_size": settings.training.test_size,
        })

        clf.fit(X_train, y_train)

        r2 = clf.score(X_test, y_test)
        print(f"R^2 on test set: {r2:.3f}")

        mlflow.log_metric("r2", float(r2))

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            X_test_path = tmpdir_path / "X_test.parquet"
            y_test_path = tmpdir_path / "y_test.parquet"

            X_test.to_parquet(X_test_path)
            y_test.to_frame("win").to_parquet(y_test_path)

            mlflow.log_artifact(str(X_test_path), artifact_path="eval_data")
            mlflow.log_artifact(str(y_test_path), artifact_path="eval_data")

        signature = infer_signature(X_train, clf.predict(X_train))

        mlflow.sklearn.log_model(
            clf,
            name="model",
            signature=signature,
            input_example=X_train.head(5),
        )
