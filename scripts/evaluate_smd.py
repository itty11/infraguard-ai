"""Evaluate detectors on the REAL Server Machine Dataset (SMD, Su et al., KDD 2019).

Setup:  git clone --depth 1 https://github.com/NetManAIOps/OmniAnomaly.git data/external/OmniAnomaly
Run:    python scripts/evaluate_smd.py --quick        (4 machines)
        python scripts/evaluate_smd.py               (all 28 machines)

Protocol per machine: fit on the first 80% of the train half, set each detector's threshold from the
last 20% (chronological), then score the untouched test half. Events/metrics as in infraguard.evaluation.
SMD has no timestamps or incident ids, so grouping metrics and the hour-of-day baseline do not apply.
"""
import _path  # noqa: F401
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler

from infraguard.config import ARTIFACT_DIR, DATA_DIR, THRESHOLD_QUANTILE
from infraguard.evaluation import event_counts, summarize_counts
from infraguard.models import AutoencoderDetector, IsolationForestDetector

QUICK = ["machine-1-1", "machine-2-1", "machine-3-1", "machine-3-7"]
p = argparse.ArgumentParser()
p.add_argument("--dir", default=str(DATA_DIR / "external" / "OmniAnomaly" / "ServerMachineDataset"))
p.add_argument("--quick", action="store_true", help="evaluate 4 representative machines only")
p.add_argument("--machines", default="", help="comma-separated machine names (overrides --quick)")
p.add_argument("--features", choices=["raw", "rolling"], default="rolling")
p.add_argument("--quantile", type=float, default=THRESHOLD_QUANTILE)
p.add_argument("--ae-epochs", type=int, default=20)
a = p.parse_args()
base = Path(a.dir)
if not (base / "train").exists():
    raise SystemExit(f"SMD not found at {base}. See the setup line at the top of this file.")
machines = ([m.strip() for m in a.machines.split(",") if m.strip()] or (QUICK if a.quick else None)
            or sorted(f.stem for f in (base / "train").glob("machine-*.txt")))


def make_features(x: np.ndarray) -> np.ndarray:
    if a.features == "raw":
        return x
    df = pd.DataFrame(x)
    return np.hstack([x, df.rolling(5, min_periods=1).mean().to_numpy(), df.diff().fillna(0).to_numpy()])


class ZScore:
    """Global z-score baseline: largest absolute deviation across dimensions."""
    def fit(self, X):
        self.m, self.s = X.mean(0), np.where(X.std(0) < 1e-9, 1.0, X.std(0))
        return self

    def score(self, X):
        return np.abs((X - self.m) / self.s).max(1)


class PCARecon:
    """Linear baseline: PCA reconstruction error (keeps 95% of variance)."""
    def fit(self, X):
        self.pca = PCA(n_components=0.95, svd_solver="full").fit(X)
        return self

    def score(self, X):
        return ((X - self.pca.inverse_transform(self.pca.transform(X))) ** 2).mean(1)


DETECTORS = {
    "zscore_global": lambda: ZScore(),
    "pca_reconstruction": lambda: PCARecon(),
    "isolation_forest": lambda: IsolationForestDetector(),
    "autoencoder": lambda: AutoencoderDetector(epochs=a.ae_epochs),
}

counts = {k: [] for k in DETECTORS}
ap = {k: [] for k in DETECTORS}
per_machine = []
t0 = time.time()
for m in machines:
    train = np.loadtxt(base / "train" / f"{m}.txt", delimiter=",")
    test = np.loadtxt(base / "test" / f"{m}.txt", delimiter=",")
    y = np.loadtxt(base / "test_label" / f"{m}.txt").astype(int)
    cut = int(len(train) * 0.8)
    ftr, fte = make_features(train), make_features(test)
    scaler = StandardScaler().fit(ftr[:cut])
    x_fit, x_val, x_te = (scaler.transform(v) for v in (ftr[:cut], ftr[cut:], fte))
    for name, factory in DETECTORS.items():
        det = factory().fit(x_fit)
        thr = float(np.quantile(det.score(x_val), a.quantile))
        s = det.score(x_te)
        c = event_counts(y, s > thr)
        counts[name].append(c)
        ap[name].append(float(average_precision_score(y, s)) if 0 < y.sum() < len(y) else np.nan)
        per_machine.append({"machine": m, "detector": name, "events": c["n_gt"], "detected": c["detected"],
                            "predicted": c["n_pred"], "false_alerts": c["fp"]})
    print(f"{m}: done ({time.time() - t0:.0f}s)", flush=True)

report = {}
for name in DETECTORS:
    row = summarize_counts(counts[name])
    row["mean_pointwise_average_precision"] = round(float(np.nanmean(ap[name])), 3)
    report[name] = row
report["_meta"] = {"machines": len(machines), "features": a.features, "quantile": a.quantile,
                   "autoencoder_backend": getattr(DETECTORS["autoencoder"]().fit(np.random.rand(200, 5)), "backend_", "?")}
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
(ARTIFACT_DIR / "smd_report.json").write_text(json.dumps({"summary": report, "per_machine": per_machine}, indent=2))
print(f"\nSMD results: {len(machines)} machines, features={a.features}, quantile={a.quantile}\n")
print(pd.DataFrame({k: v for k, v in report.items() if k != "_meta"}).T.to_string())
print("\nautoencoder backend:", report["_meta"]["autoencoder_backend"])