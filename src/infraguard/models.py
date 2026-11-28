"""Anomaly detectors: hour-of-day z-score baseline, Isolation Forest, Autoencoder."""
from __future__ import annotations

import os
import tempfile

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neural_network import MLPRegressor


class HourlyZBaseline:
    """Per-(server, hour-of-day) mean/std of normal behaviour.

    Used (a) as the simple statistical baseline detector and (b) to attribute which
    metrics are abnormal inside an incident (Isolation Forest has no native attribution).
    """

    def __init__(self, cols: list[str]):
        self.cols = list(cols)

    def fit(self, df: pd.DataFrame) -> "HourlyZBaseline":
        d = df.assign(_hour=df["timestamp"].dt.hour)
        grp = d.groupby(["server_id", "_hour"])[self.cols]
        self.mean_, self.std_ = grp.mean(), grp.std()
        self.gmean_, self.gstd_ = d[self.cols].mean(), d[self.cols].std()
        return self

    def zscores(self, df: pd.DataFrame) -> np.ndarray:
        keys = pd.MultiIndex.from_arrays([df["server_id"], df["timestamp"].dt.hour])
        mean = self.mean_.reindex(keys).fillna(self.gmean_).to_numpy()
        std = self.std_.reindex(keys).fillna(self.gstd_).to_numpy()
        std = np.maximum(std, 0.03 * np.abs(mean) + 1e-3)
        return (df[self.cols].to_numpy() - mean) / std


class IsolationForestDetector:
    name = "isolation_forest"

    def __init__(self, n_estimators: int = 200, random_state: int = 42):
        self.model = IsolationForest(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)

    def fit(self, X: np.ndarray) -> "IsolationForestDetector":
        self.model.fit(X)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        return -self.model.score_samples(X)   # higher = more anomalous


class AutoencoderDetector:
    """Reconstruction-error detector: score = mean((x - x_hat)^2).

    Uses Keras when TensorFlow is installed, otherwise falls back to an sklearn MLP
    trained to reconstruct its input (same idea, no deep-learning dependency).
    """
    name = "autoencoder"

    def __init__(self, epochs: int = 30, batch_size: int = 256, seed: int = 42, backend: str = "auto"):
        self.epochs, self.batch_size, self.seed, self.backend = epochs, batch_size, seed, backend
        self.model_ = None

    @staticmethod
    def _keras():
        try:
            os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
            from tensorflow import keras  # type: ignore
            return keras
        except Exception:
            return None

    def fit(self, X: np.ndarray) -> "AutoencoderDetector":
        keras = self._keras() if self.backend in ("auto", "keras") else None
        if keras is not None:
            keras.utils.set_random_seed(self.seed)
            dim = X.shape[1]
            m = keras.Sequential([
                keras.layers.Input(shape=(dim,)),
                keras.layers.Dense(32, activation="relu"),
                keras.layers.Dense(12, activation="relu"),   # bottleneck
                keras.layers.Dense(32, activation="relu"),
                keras.layers.Dense(dim),
            ])
            m.compile(optimizer="adam", loss="mse")
            m.fit(X, X, epochs=self.epochs, batch_size=self.batch_size, validation_split=0.1, verbose=0,
                  callbacks=[keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)])
            self.model_, self.backend_ = m, "keras"
        else:
            m = MLPRegressor(hidden_layer_sizes=(32, 12, 32), max_iter=self.epochs * 10, early_stopping=True,
                             random_state=self.seed)
            m.fit(X, X)
            self.model_, self.backend_ = m, "sklearn"
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError(
                "This bundle's autoencoder was trained with TensorFlow/Keras, which is not installed here. "
                "Install requirements-dl.txt, or retrain in this environment (python scripts/train.py).")
        recon = self.model_.predict(X, verbose=0) if self.backend_ == "keras" else self.model_.predict(X)
        return np.mean((X - recon) ** 2, axis=1)

    # Keras models are not picklable; embed the serialized model so joblib can store it.
    def __getstate__(self):
        state = self.__dict__.copy()
        if state.get("backend_") == "keras":
            with tempfile.TemporaryDirectory() as d:
                p = os.path.join(d, "ae.keras")
                self.model_.save(p)
                with open(p, "rb") as fh:
                    state["model_"] = fh.read()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if state.get("backend_") == "keras":
            keras = self._keras()
            if keras is None:          # bundle trained with Keras, but TensorFlow is missing in this environment
                self.model_ = None     # other detectors keep working; scoring with this one raises a clear error
                return
            with tempfile.TemporaryDirectory() as d:
                p = os.path.join(d, "ae.keras")
                with open(p, "wb") as fh:
                    fh.write(state["model_"])
                self.model_ = keras.models.load_model(p)