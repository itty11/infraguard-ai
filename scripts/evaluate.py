"""Compare all detectors on the held-out labelled test set and write artifacts/eval_report.json."""
import _path  # noqa: F401
import json

import pandas as pd

from infraguard.config import ARTIFACT_DIR, DATA_DIR
from infraguard.evaluation import evaluate_run
from infraguard.pipeline import MODEL_NAMES, DetectorBundle, detect

bundle = DetectorBundle.load(ARTIFACT_DIR)
test = pd.read_csv(DATA_DIR / "test.csv")
report = {}
for name in MODEL_NAMES:
    feats, _, incidents = detect(bundle, test, model=name)
    report[name] = evaluate_run(feats, incidents)
    report[name]["threshold"] = round(bundle.thresholds[name], 4)

(ARTIFACT_DIR / "eval_report.json").write_text(json.dumps(report, indent=2))
cols = ["event_precision", "event_recall", "event_f1", "false_alerts_per_day_per_server",
        "median_detection_latency_min", "pointwise_average_precision", "incident_grouping_ari",
        "alert_reduction_ratio"]
print(pd.DataFrame({m: {c: r[c] for c in cols} for m, r in report.items()}).T.to_string())
print("\nrecall by anomaly type:")
print(pd.DataFrame({m: r["recall_by_type"] for m, r in report.items()}).T.to_string())
