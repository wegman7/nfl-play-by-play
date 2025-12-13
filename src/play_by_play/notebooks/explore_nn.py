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

# %%
# train/test split
X = clean_df[settings.schema.numeric_features + settings.schema.categorical_features]
y = clean_df[settings.schema.label_cols[0]]
keys = clean_df[settings.schema.key_cols]

X_train, X_test, y_train, y_test, keys_train, keys_test = train_test_split(
    X, y, keys, test_size=0.2, random_state=42
)

# %%
import numpy as np
import tensorflow as tf
from scipy import sparse

# Fit + transform
preprocessor.fit(X_train)
X_train_p = preprocessor.transform(X_train)
X_test_p  = preprocessor.transform(X_test)

print(type(X_train_p), getattr(X_train_p, "shape", None))

def to_tf_input(X):
    # SciPy sparse -> TF SparseTensor
    if sparse.issparse(X):
        X = X.tocoo()
        indices = np.vstack([X.row, X.col]).T.astype(np.int64)
        values = X.data.astype(np.float32)
        shape = np.array(X.shape, dtype=np.int64)
        return tf.sparse.reorder(tf.SparseTensor(indices, values, shape)), True

    # Dense numpy -> TF dense tensor
    X = np.asarray(X, dtype=np.float32)
    return X, False

X_train_tf, is_sparse = to_tf_input(X_train_p)
X_test_tf, _ = to_tf_input(X_test_p)

print("sparse?", is_sparse, "shape:", (X_train_p.shape))



# %%
tf.random.set_seed(42)

inp = tf.keras.Input(shape=(X_train_p.shape[1],), sparse=is_sparse)
x = tf.keras.layers.Dense(32, activation="relu")(inp)
x = tf.keras.layers.Dense(16, activation="relu")(x)
out = tf.keras.layers.Dense(1, activation="sigmoid")(x)

nn = tf.keras.Model(inp, out)
nn.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="binary_crossentropy",
    metrics=[tf.keras.metrics.AUC(name="auc"), tf.keras.metrics.BinaryAccuracy(name="acc")],
)

history = nn.fit(
    X_train_tf, y_train.to_numpy(),
    validation_split=0.2,
    epochs=10,
    batch_size=2048,
    verbose=1,
)


# %%
# predict
# Probabilities (0..1)
p_test = nn.predict(X_test_p).ravel()

# If you want hard labels:
y_hat = (p_test >= 0.5).astype(int)

# Quick metrics
from sklearn.metrics import log_loss, roc_auc_score, accuracy_score

print("log_loss:", log_loss(y_test, p_test))
print("auc:", roc_auc_score(y_test, p_test))
print("acc:", accuracy_score(y_test, y_hat))

analyze = (
    X_test
    .assign(prediction=p_test, actual=y_test)
    .join(keys_test)
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
ten_games
# game = (
#     ten_games
#     .groupby('game_id')['game_id'].count()
# )
# game

# %%
game = analyze[analyze['game_id'] == '2025_05_TEN_ARI'].sort_values('time_seconds_total', ascending=False)

plt.plot(game['time_seconds_total'], game['prediction'], label='Prediction')
plt.gca().invert_xaxis()
plt.show()

# LOOK AT DIFFERENT ARCHITECTURES - DOES RANDOM FOREST WORK BETTER THAN NN?
# START WITH ONE FEATURE AND INCREASE
# HOW EXACTLY DOES RANDOM FOREST WORK??
# %%
