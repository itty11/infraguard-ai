"""Train all detectors on normal data and save the bundle."""
import _path  # noqa: F401
import argparse
import time

import pandas as pd

from infraguard.config import ARTIFACT_DIR, DATA_DIR
from infraguard.pipeline import DetectorBundle

p = argparse.ArgumentParser()
p.add_argument("--epochs", type=int, default=30, help="autoencoder epochs")
a = p.parse_args()

t0 = time.time()
train, val = pd.read_csv(DATA_DIR / "train.csv"), pd.read_csv(DATA_DIR / "val.csv")
bundle = DetectorBundle().fit(train, val, ae_epochs=a.epochs)
path = bundle.save(ARTIFACT_DIR)
print(f"saved {path} in {time.time() - t0:.1f}s")
print("autoencoder backend:", bundle.models["autoencoder"].backend_)
print("thresholds:", {k: round(v, 4) for k, v in bundle.thresholds.items()})
