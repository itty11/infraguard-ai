"""Event-based evaluation (point-wise F1 is misleading for time series)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, average_precision_score

TOL = pd.Timedelta(minutes=5)


def ground_truth_events(feats: pd.DataFrame) -> pd.DataFrame:
    t = feats[feats["true_incident_id"] > 0]
    return (t.groupby("true_incident_id")
            .agg(server_id=("server_id", "first"), start=("timestamp", "min"),
                 end=("timestamp", "max"), type=("anomaly_type", "first")).reset_index())


def evaluate_run(feats: pd.DataFrame, incidents: list[dict]) -> dict:
    gt = ground_truth_events(feats)
    days = (feats["timestamp"].max() - feats["timestamp"].min()).total_seconds() / 86400
    n_servers = feats["server_id"].nunique()

    def overlaps(inc, ev):
        return (inc["server_id"] == ev.server_id and inc["start"] <= ev.end + TOL
                and inc["end"] >= ev.start - TOL)

    detected, latencies, per_type = [], [], {}
    for ev in gt.itertuples():
        hits = [i for i in incidents if overlaps(i, ev)]
        detected.append(bool(hits))
        per_type.setdefault(ev.type, []).append(bool(hits))
        if hits:
            first = min(h["detected_at"] for h in hits)
            latencies.append(max(0.0, (first - ev.start).total_seconds() / 60))
    tp_pred = sum(any(overlaps(i, ev) for ev in gt.itertuples()) for i in incidents)
    n_pred, n_gt = len(incidents), len(gt)
    precision = tp_pred / n_pred if n_pred else 0.0
    recall = float(np.mean(detected)) if n_gt else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    pred_id = np.zeros(len(feats), dtype=int)
    for k, inc in enumerate(incidents, 1):
        m = (feats["server_id"] == inc["server_id"]) & feats["timestamp"].between(inc["start"], inc["end"])
        pred_id[m.to_numpy()] = k
    true_id = feats["true_incident_id"].to_numpy()
    sel = (pred_id > 0) | (true_id > 0)
    ari = float(adjusted_rand_score(true_id[sel], pred_id[sel])) if sel.sum() > 1 else None
    raw_alerts = sum(len(i["anomalous_metrics"]) for i in incidents)
    y = feats["is_anomaly"].to_numpy()
    ap = float(average_precision_score(y, feats["score"])) if 0 < y.sum() < len(y) else None
    return {
        "events_true": n_gt, "incidents_predicted": n_pred,
        "event_precision": round(precision, 3), "event_recall": round(recall, 3), "event_f1": round(f1, 3),
        "false_alerts_per_day_per_server": round((n_pred - tp_pred) / max(days * n_servers, 1e-9), 3),
        "median_detection_latency_min": round(float(np.median(latencies)), 1) if latencies else None,
        "pointwise_average_precision": round(ap, 3) if ap is not None else None,
        "incident_grouping_ari": round(ari, 3) if ari is not None else None,
        "alert_reduction_ratio": round(raw_alerts / n_pred, 2) if n_pred else None,
        "recall_by_type": {k: round(float(np.mean(v)), 2) for k, v in sorted(per_type.items())},
    }
