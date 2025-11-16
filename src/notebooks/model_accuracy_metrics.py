# %%
import sys
from pathlib import Path

import mlflow
import mlflow.sklearn  # you'll need this for load_model

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

# Use the same tracking backend as training
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("play_by_play_win_prob")  # optional but harmless

# %%
from mlflow.tracking import MlflowClient

client = MlflowClient()
experiment = mlflow.get_experiment_by_name("play_by_play_win_prob")
if experiment is None:
    raise RuntimeError("Experiment 'play_by_play_win_prob' not found in this tracking URI")

runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["metrics.r2 DESC"],
    max_results=1,
)

best_run = runs[0]
RUN_ID = best_run.info.run_id
model_uri = f"runs:/{RUN_ID}/model"

clf = mlflow.sklearn.load_model(model_uri)
print("Loaded model from:", model_uri)


# # %%
# y_pred = clf.predict(X_test)
# pred_series = pd.Series(y_pred, index=X_test.index, name="win_pred")

# df_with_preds = X_test.join(pred_series, how="left")
# df_with_preds["win_true"] = y_test
# %%
