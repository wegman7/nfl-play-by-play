# %%
from pathlib import Path
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT))

from src.play_by_play.ml.features import build_features
from src.play_by_play.ml.labels import build_labels
from src.play_by_play.ml.util import convert_posteam_to_home_pred
from src.play_by_play.config.settings import settings

# %%
data_dir = ROOT / "data" / "raw"
full_df = pd.DataFrame()
for i in range(1999, 2026):
    path = data_dir / f"play_by_play_{i}.parquet"
    df = pd.read_parquet(path)
    print(f"Year: {i}, Shape: {df.shape}")
    full_df = pd.concat([full_df, df], axis=0, ignore_index=True)

# %%
features = build_features(full_df)
labels = build_labels(full_df)
clean_df = pd.merge(features, labels, on=["game_id", "play_id"])

# %%
numeric_transformer = "passthrough"

categorical_transformer = OneHotEncoder(
    handle_unknown="ignore",  # avoids crashing on unseen teams/locations
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, settings.schema.numeric_features),
        ("cat", categorical_transformer, settings.schema.categorical_features),
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
X = clean_df[settings.schema.numeric_features + settings.schema.categorical_features]
y = clean_df[settings.schema.label_cols[0]]
keys = clean_df[settings.schema.key_cols]

X_train, X_test, y_train, y_test, keys_train, keys_test = train_test_split(
    X, y, keys, test_size=0.2, random_state=42
)


# %%
 # Fit the model
clf.fit(X_train, y_train)

# %%
# Evaluate quickly
r2 = clf.score(X_test, y_test)
print(f"R^2 on test set: {r2:.3f}")

# evaluate mean squared error
from sklearn.metrics import mean_squared_error
y_pred = clf.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"MSE on test set: {mse:.3f}")

# %%
analyze = X_test.copy()
analyze['prediction'] = y_pred
analyze['prediction_home'] = convert_posteam_to_home_pred(analyze)
analyze['actual'] = y_test
analyze = analyze.join(keys_test)
analyze['correct'] = np.select(
    [(analyze['prediction'] >= .5) & (analyze['actual'] >= .5),
     (analyze['prediction'] < .5) & (analyze['actual'] < .5)],
    [1, 1],
    default=0
)


# %%
# graph accuracy over for different time buckets in the game
import matplotlib.pyplot as plt
analyze["time_bucket"] = pd.cut(analyze["time_seconds_total"], bins=20)

analyze.groupby("time_bucket")["correct"].mean().plot(
    title="Prediction Accuracy over Time (Binned)"
)
plt.gca().invert_xaxis()
plt.show()

# %%
ten_games = analyze[analyze['game_id'].str.contains('TEN')]['game_id'].drop_duplicates().sort_values()

# %%
game = analyze[analyze['game_id'] == '2025_05_TEN_ARI'].sort_values('time_seconds_total', ascending=False)

plt.plot(game['time_seconds_total'], game['prediction_home'], label='Prediction')
plt.gca().invert_xaxis()
plt.show()

# LOOK AT DIFFERENT ARCHITECTURES - DOES RANDOM FOREST WORK BETTER THAN NN?
# START WITH ONE FEATURE AND INCREASE
# HOW EXACTLY DOES RANDOM FOREST WORK??
# %%
import shap

# Get transformed feature names after preprocessing
preprocessor_fitted = clf.named_steps['preprocessor']
cat_features = preprocessor_fitted.named_transformers_['cat'].get_feature_names_out(
    settings.schema.categorical_features
)
all_feature_names = list(settings.schema.numeric_features) + list(cat_features)

# Transform the test set to get preprocessed features
X_test_transformed = preprocessor_fitted.transform(X_test)

# Create SHAP explainer for the Random Forest model
rf_model = clf.named_steps['model']
explainer = shap.TreeExplainer(rf_model)

# Calculate SHAP values (use a sample if dataset is large)
sample_size = min(1000, len(X_test_transformed))
X_sample = X_test_transformed[:sample_size]
shap_values = explainer.shap_values(X_sample)

# %%
# Summary plot - shows feature importance
shap.summary_plot(shap_values, X_sample, feature_names=all_feature_names, show=False)
plt.tight_layout()
plt.show()

# %%
# Bar plot - average impact on model output
shap.summary_plot(shap_values, X_sample, feature_names=all_feature_names, plot_type="bar", show=False)
plt.tight_layout()
plt.show()

# %%
# %%
# --- SHAP (fast-ish) ---
# pip install shap

import shap

# Pull fitted pieces out of the pipeline
pre = clf.named_steps["preprocessor"]
rf = clf.named_steps["model"]

# Feature names after preprocessing (works on sklearn >= 1.0+)
try:
    feature_names = pre.get_feature_names_out()
except Exception as e:
    raise RuntimeError(
        "pre.get_feature_names_out() failed. "
        "Upgrade scikit-learn or share the error and I'll give a robust fallback."
    ) from e

# Use a SMALL sample so this doesn't take forever
N_SHAP = 200  # bump to 500 later if you want; start small
rng = np.random.RandomState(42)
idx = rng.choice(X_test.shape[0], size=min(N_SHAP, X_test.shape[0]), replace=False)

X_shap = X_test.iloc[idx]
X_shap_t = pre.transform(X_shap)

# TreeExplainer is the fast one for RandomForest
explainer = shap.TreeExplainer(
    rf,
    feature_perturbation="tree_path_dependent",
)

shap_values = explainer.shap_values(X_shap_t)  # (n_samples, n_features) for regressor

# Quick numeric importance (fast, no plots)
mean_abs_shap = np.abs(shap_values).mean(axis=0)
imp = (
    pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs_shap})
    .sort_values("mean_abs_shap", ascending=False)
)
print("\nTop SHAP features (mean |SHAP|):")
print(imp.head(20).to_string(index=False))

# Optional: one plot (can be slower; comment out if you want)
# shap.summary_plot(shap_values, X_shap_t, feature_names=feature_names, plot_type="bar", show=True)

# --- Local explanation for a specific play (should be quick) ---
target_game_id = "2025_05_TEN_ARI"
target_play_id = 1504.0

row = analyze[(analyze["game_id"] == target_game_id) & (analyze["play_id"] == target_play_id)]
if len(row) == 1:
    x1 = row[settings.schema.numeric_features + settings.schema.categorical_features]
    x1_t = pre.transform(x1)

    sv1 = explainer.shap_values(x1_t)[0]
    base = float(explainer.expected_value)
    pred = float(rf.predict(x1_t)[0])

    print(f"\nLocal SHAP for {target_game_id} play_id={target_play_id}")
    print(f"base={base:.6f}  pred={pred:.6f}  base+sum(shap)={base + float(sv1.sum()):.6f}")

    top_k = 20
    order = np.argsort(np.abs(sv1))[::-1][:top_k]
    print("\nTop contributors:")
    for j in order:
        print(f"{feature_names[j]:45s}  shap={sv1[j]: .6f}")
else:
    print(f"\nCouldn't find exactly one row for game_id={target_game_id}, play_id={target_play_id}. Found {len(row)} rows.")
