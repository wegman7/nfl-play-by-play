# %%
from pathlib import Path
import sys
import tensorflow as tf
import pandas as pd
import numpy as np
import os

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT))

from src.play_by_play.ml.features import build_features
from src.play_by_play.ml.labels import build_labels
from src.play_by_play.config.settings import settings

print("TensorFlow version:", tf.__version__)
print("GPU available:", len(tf.config.list_physical_devices('GPU')) > 0)

# %%
data_dir = ROOT / "data" / "raw"
full_df = pd.DataFrame()
for i in range(2025, 2026):
    path = data_dir / f"play_by_play_{i}.parquet"
    df = pd.read_parquet(path)
    print(f"Year: {i}, Shape: {df.shape}")
    full_df = pd.concat([full_df, df], axis=0, ignore_index=True)

# %%
features = build_features(full_df)
labels = build_labels(full_df)
clean_df = pd.merge(features, labels, on=["game_id", "play_id"])

# %%
preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", settings.schema.numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), settings.schema.categorical_features),
    ]
)

# %%
X = clean_df[settings.schema.numeric_features + settings.schema.categorical_features]
y = clean_df[settings.schema.label_cols[0]]
keys = clean_df[settings.schema.key_cols]

# Train/test split
X_train, X_test, y_train, y_test, keys_train, keys_test = train_test_split(
    X, y, keys,
    test_size=0.2,
    random_state=42
)

# %%
print("Fitting preprocessor...")
preprocessor.fit(X_train)

print("Transforming train data...")
X_train_p = preprocessor.transform(X_train)

print("Transforming test data...")
X_test_p = preprocessor.transform(X_test)

# Convert to dense if sparse
from scipy import sparse
if sparse.issparse(X_train_p):
    print("Converting to dense arrays...")
    X_train_p = X_train_p.toarray()
    X_test_p = X_test_p.toarray()

print(f"Training shape: {X_train_p.shape}, Test shape: {X_test_p.shape}")

# %%
tf.random.set_seed(42)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation="relu", input_shape=(X_train_p.shape[1],)),
    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
)

history = model.fit(
    X_train_p, y_train.to_numpy(),
    validation_split=0.2,
    epochs=10,
    batch_size=4096,
    verbose=1,
)

# %%
predictions = model.predict(X_test_p).ravel()

y_test_array = y_test.to_numpy()
# Calculate binary cross-entropy manually (same as log_loss for binary classification)
bce = -np.mean(y_test_array * np.log(predictions + 1e-15) + (1 - y_test_array) * np.log(1 - predictions + 1e-15))
print("Binary Cross-Entropy:", bce)

# For AUC, filter out ties (0.5 values) to get a true binary classification metric
non_tie_mask = (y_test_array != 0.5)
if non_tie_mask.sum() > 0:
    print("AUC (excluding ties):", roc_auc_score(y_test_array[non_tie_mask], predictions[non_tie_mask]))
else:
    print("AUC: N/A (all ties)")

# For accuracy, convert y_test to binary (>= 0.5 is 1, < 0.5 is 0)
y_test_binary = (y_test_array >= 0.5).astype(int)
print("Accuracy:", accuracy_score(y_test_binary, (predictions >= 0.5).astype(int)))

# %%
analyze = (
    X_test
    .assign(prediction=predictions, actual=y_test)
    .join(keys_test)
)

# %%
analyze["time_bucket"] = pd.cut(analyze["time_seconds_total"], bins=20)
analyze["correct"] = ((analyze["prediction"] >= 0.5) == (analyze["actual"] >= 0.5)).astype(int)

analyze.groupby("time_bucket")["correct"].mean().plot(
    title="Prediction Accuracy over Time (Binned)"
)
plt.gca().invert_xaxis()
plt.show()

# %%
game = analyze[analyze['game_id'] == '2025_05_TEN_ARI'].sort_values('time_seconds_total', ascending=False)

plt.plot(game['time_seconds_total'], game['prediction'], label='Prediction')
plt.gca().invert_xaxis()
plt.show()

# %%
