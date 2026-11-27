"""Training bundle + end-to-end detection (clean -> features -> score -> flags -> incidents)."""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .config import (ACTIVE_MODEL, FEATURE_COLUMNS, THRESHOLD_QUANTILE, WARMUP, Z_COLUMNS)
from .correlation import correlate
from .features import build_features, clean
from .models import AutoencoderDetector, HourlyZBaseline, IsolationForestDetector

MODEL_NAMES = ["static_threshold", "zscore_baseline", "isolation_forest", "autoencoder"]


def static_score(f: pd.DataFrame) -> np.ndarray:
    """Classic fixed-threshold monitoring: CPU>90%, memory>90%, latency>200ms, loss>5%. Score>1 = alert."""
    return np.max(np.vstack([f["cpu_usage"] / 90, f["memory_usage"] / 90,
                             f["latency"] / 200, f["packet_loss"] / 5]), axis=0)


class DetectorBundle:
    def __init__(self, quantile: float = THRESHOLD_QUANTILE, active: str = ACTIVE_MODEL):
        self.quantile, self.active = quantile, active
        self.models: dict = {}
        self.thresholds: dict[str, float] = {}

    def fit(self, train_raw: pd.DataFrame, val_raw: pd.DataFrame, ae_epochs: int = 30) -> "DetectorBundle":
        """Train on NORMAL data only; thresholds come from normal validation data."""
        tr, va = build_features(clean(train_raw)), build_features(clean(val_raw))
        self.scaler = StandardScaler().fit(tr[FEATURE_COLUMNS])
        self.baseline = HourlyZBaseline(Z_COLUMNS).fit(tr)
        x_tr = self.scaler.transform(tr[FEATURE_COLUMNS])
        self.models["isolation_forest"] = IsolationForestDetector().fit(x_tr)
        self.models["autoencoder"] = AutoencoderDetector(epochs=ae_epochs).fit(x_tr)
        self.thresholds["static_threshold"] = 1.0
        for name in ("zscore_baseline", "isolation_forest", "autoencoder"):
            self.thresholds[name] = float(np.quantile(self.score(name, va), self.quantile))
        return self

    def score(self, name: str, feats: pd.DataFrame) -> np.ndarray:
        if name == "static_threshold":
            return static_score(feats)
        if name == "zscore_baseline":
            return np.abs(self.baseline.zscores(feats)).max(axis=1)
        return self.models[name].score(self.scaler.transform(feats[FEATURE_COLUMNS]))

    def save(self, directory: Path) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "bundle.joblib"
        joblib.dump(self, path)
        return path

    @staticmethod
    def load(directory: Path) -> "DetectorBundle":
        return joblib.load(Path(directory) / "bundle.joblib")


def detect(bundle: DetectorBundle, raw: pd.DataFrame, model: str | None = None):
    """Run detection on raw telemetry (any number of servers). Returns (feats, z, incidents)."""
    model = model or bundle.active
    feats = build_features(clean(raw))
    feats["score"] = bundle.score(model, feats)
    feats["threshold"] = bundle.thresholds[model]
    feats["flag"] = feats["score"] > feats["threshold"]
    feats.loc[feats.groupby("server_id").cumcount() < WARMUP, "flag"] = False
    z = pd.DataFrame(bundle.baseline.zscores(feats), columns=Z_COLUMNS, index=feats.index)
    return feats, z, correlate(feats, z, model)
