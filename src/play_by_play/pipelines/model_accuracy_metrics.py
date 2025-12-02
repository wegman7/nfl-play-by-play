from pathlib import Path

import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error

import mlflow
import mlflow.sklearn
from mlflow.artifacts import download_artifacts

from play_by_play.utils.model_util import load_best_model_from_experiment


def run_model_accuracy_metrics():
    TRACKING_URI = "sqlite:///mlflow.db"
    EXPERIMENT_NAME = "play_by_play_win_prob"

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    clf, model_uri, RUN_ID = load_best_model_from_experiment(
        experiment_name=EXPERIMENT_NAME,
        tracking_uri=TRACKING_URI,
        metric="r2",
        higher_is_better=True,
    )

    print("Using run:", RUN_ID)
    print("Loaded model from:", model_uri)

    local_dir = download_artifacts(
        run_id=RUN_ID,
        artifact_path="eval_data",
    )

    local_dir = Path(local_dir)
    X_test = pd.read_parquet(local_dir / "X_test.parquet")
    y_test = pd.read_parquet(local_dir / "y_test.parquet")["win"]

    print("Loaded X_test shape:", X_test.shape)
    print("Loaded y_test shape:", y_test.shape)

    y_pred = clf.predict(X_test)

    df_with_preds = X_test.copy()
    df_with_preds["win_true"] = y_test.values
    df_with_preds["win_pred"] = y_pred

    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    print(f"R^2 (recomputed on artifact test set): {r2:.4f}")
    print(f"MSE: {mse:.6f}")
