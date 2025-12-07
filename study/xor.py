# %%
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

np.random.seed(42)
tf.random.set_seed(42)

def make_continuous_xor(n_per_quad=200, noise=0.15):
    """
    Generate a 'continuous' XOR dataset:
    four Gaussian blobs around (0,0), (0,1), (1,0), (1,1)
    with labels following XOR.
    """
    centers = np.array([
        [0.0, 0.0],  # label 0
        [0.0, 1.0],  # label 1
        [1.0, 0.0],  # label 1
        [1.0, 1.0]   # label 0
    ])
    labels = np.array([0, 1, 1, 0])

    X = []
    y = []
    for c, lab in zip(centers, labels):
        X.append(c + noise * np.random.randn(n_per_quad, 2))
        y.append(np.full(n_per_quad, lab))

    X = np.vstack(X).astype("float32")
    y = np.concatenate(y).astype("float32")

    # Shuffle
    idx = np.random.permutation(len(X))
    return X[idx], y[idx]

X, y = make_continuous_xor(n_per_quad=100, noise=.1)

# Train / test split
split = int(0.7 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print(X_train.shape, X_test.shape)





# %%
import matplotlib.pyplot as plt


import numpy as np

def plot_xor_data(X, y, title="Continuous XOR Dataset"):
    plt.figure(figsize=(6,6))
    plt.scatter(
        X[:,0], X[:,1],
        c=y, cmap="bwr", edgecolor="k", alpha=0.7
    )
    plt.title(title)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.grid(True)
    plt.show()

plot_xor_data(X, y)







# %%
def build_overfitting_model():
    model = models.Sequential([
        layers.Input(shape=(2,)),
        layers.Dense(64, activation="tanh"),
        layers.Dense(64, activation="tanh"),
        layers.Dense(64, activation="tanh"),
        layers.Dense(1, activation="sigmoid")
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model

overfit_model = build_overfitting_model()

history_overfit = overfit_model.fit(
    X_train, y_train,
    epochs=400,                 # long training → encourages overfitting
    batch_size=16,
    validation_split=0.3,       # hold-out validation
    verbose=0
)

print("Overfit model test accuracy:",
      overfit_model.evaluate(X_test, y_test, verbose=0))









# %%
def build_regularized_model():
    model = models.Sequential([
        layers.Input(shape=(2,)),
        layers.Dense(
            12,
            activation="tanh",
            kernel_regularizer=regularizers.l2(1e-3)
        ),
        layers.Dense(
            6,
            activation="tanh",
            kernel_regularizer=regularizers.l2(1e-3)
        ),
        layers.Dense(1, activation="sigmoid")
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model

reg_model = build_regularized_model()

early_stop = tf.keras.callbacks.EarlyStopping(
    patience=30,
    restore_best_weights=True
)

history_reg = reg_model.fit(
    X_train, y_train,
    epochs=400,
    batch_size=16,
    validation_split=0.3,
    callbacks=[early_stop],
    verbose=0
)

print("Regularized model test accuracy:",
      reg_model.evaluate(X_test, y_test, verbose=0))








# %%
def plot_loss_curves(history, title):
    plt.figure(figsize=(6,4))
    plt.plot(history.history["loss"], label="train loss")
    plt.plot(history.history["val_loss"], label="val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

plot_loss_curves(history_overfit, "Overfitting model – loss")
plot_loss_curves(history_reg, "Regularized model – loss")







# %%
def plot_decision_boundary(model, X, y, title, grid_step=0.01):
    # make a grid
    x_min, x_max = X[:,0].min() - 0.5, X[:,0].max() + 0.5
    y_min, y_max = X[:,1].min() - 0.5, X[:,1].max() + 0.5
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, grid_step),
        np.arange(y_min, y_max, grid_step)
    )
    grid = np.c_[xx.ravel(), yy.ravel()].astype("float32")
    print(grid)

    # model prediction on grid
    Z = model.predict(grid, verbose=0)
    Z = Z.reshape(xx.shape)
    print(Z)

    plt.figure(figsize=(6,6))
    # probability background
    cs = plt.contourf(xx, yy, Z, levels=50, alpha=0.9, cmap="RdBu")
    plt.colorbar(cs, label="P(class=1)")

    # training points
    plt.scatter(X[:,0], X[:,1],
                c=y, cmap="bwr", edgecolor="k", alpha=0.8)

    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title(title)
    plt.show()

plot_decision_boundary(overfit_model, X_train, y_train,
                       "Overfitting model decision boundary")
plot_decision_boundary(reg_model, X_train, y_train,
                       "Regularized model decision boundary")

# %%
