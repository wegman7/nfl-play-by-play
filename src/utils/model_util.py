# src/models/mlflow_utils.py

from __future__ import annotations

from typing import Tuple

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient


def load_best_model_from_experiment(
    experiment_name: str,
    tracking_uri: str = "sqlite:///mlflow.db",
    metric: str = "r2",
    higher_is_better: bool = True,
) -> Tuple[object, str, str]:
    """
    Load the best MLflow model from an experiment based on a single metric.

    Returns:
        model: The loaded model (e.g., sklearn Pipeline)
        model_uri: The MLflow model URI (e.g. 'runs:/<run_id>/model')
        run_id: The best run's ID
    """
    mlflow.set_tracking_uri(tracking_uri)

    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise RuntimeError(f"Experiment '{experiment_name}' not found in MLflow tracking URI '{tracking_uri}'")

    client = MlflowClient()

    order = "DESC" if higher_is_better else "ASC"
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f"metrics.{metric} {order}"],
        max_results=1,
    )

    if not runs:
        raise RuntimeError(f"No runs found for experiment '{experiment_name}'")

    best_run = runs[0]
    run_id = best_run.info.run_id
    model_uri = f"runs:/{run_id}/model"

    model = mlflow.sklearn.load_model(model_uri)
    return model, model_uri, run_id
