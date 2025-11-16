# %%
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from mlflow.artifacts import download_artifacts

# %%
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

# Use the same tracking backend as training
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("play_by_play_win_prob")

# %%
# Find the best run (by r2) for this experiment
client = MlflowClient()
experiment = mlflow.get_experiment_by_name("play_by_play_win_prob")
if experiment is None:
    raise RuntimeError("Experiment 'play_by_play_win_prob' not found in this tracking URI")

runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["metrics.r2 DESC"],
    max_results=1,
)

if not runs:
    raise RuntimeError("No runs found for experiment 'play_by_play_win_prob'")

best_run = runs[0]
RUN_ID = best_run.info.run_id
print("Using run:", RUN_ID)

# %%
# Load the logged model (full sklearn Pipeline)
model_uri = f"runs:/{RUN_ID}/model"
clf = mlflow.sklearn.load_model(model_uri)
print("Loaded model from:", model_uri)

# %%
# Download X_test / y_test artifacts from this run
local_dir = download_artifacts(
    run_id=RUN_ID,
    artifact_path="eval_data",  # must match the artifact_path used in train notebook
)

local_dir = Path(local_dir)
X_test = pd.read_parquet(local_dir / "X_test.parquet")
y_test = pd.read_parquet(local_dir / "y_test.parquet")["win"]

print("Loaded X_test shape:", X_test.shape)
print("Loaded y_test shape:", y_test.shape)

# %%
# Make predictions
y_pred = clf.predict(X_test)

# Build DataFrame with predictions + truth
df_with_preds = X_test.copy()
df_with_preds["win_true"] = y_test.values
df_with_preds["win_pred"] = y_pred

# %%
# Basic evaluation metrics
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f"R^2 (recomputed on artifact test set): {r2:.4f}")
print(f"MSE: {mse:.6f}")

# df_with_preds.head()  # uncomment to inspect in a notebook
