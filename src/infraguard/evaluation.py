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


def event_counts(y_true, flags, tol: int = 5, min_consecutive: int | None = None,
                 merge_gap: int | None = None, samples_per_day: float = 1440.0) -> dict:
    """Event-level counts for ONE labelled series (e.g. one SMD machine; 1 sample = 1 minute).

    Ground-truth events are runs of consecutive 1s. Predicted events are runs of flagged samples that persist
    >= min_consecutive and are merged across gaps <= merge_gap. A true event is detected if any predicted event
    overlaps it (+/- tol samples). Counts are returned (not ratios) so several series can be pooled.
    """
    from .config import MERGE_GAP, MIN_CONSECUTIVE
    from .correlation import find_runs
    min_consecutive = MIN_CONSECUTIVE if min_consecutive is None else min_consecutive
    merge_gap = MERGE_GAP if merge_gap is None else merge_gap
    y, f = np.asarray(y_true).astype(bool), np.asarray(flags).astype(bool)
    gt, pred = find_runs(y, 1, 0), find_runs(f, min_consecutive, merge_gap)

    def overlap(p, g):
        return p[0] <= g[1] + tol and p[1] >= g[0] - tol

    latencies, detected = [], 0
    for g in gt:
        hits = [p for p in pred if overlap(p, g)]
        if hits:
            detected += 1
            first_alert = min(p[0] for p in hits) + min_consecutive - 1
            latencies.append(max(0, first_alert - g[0]))
    tp_pred = sum(any(overlap(p, g) for g in gt) for p in pred)
    return {"n_gt": len(gt), "detected": detected, "n_pred": len(pred), "tp_pred": tp_pred,
            "fp": len(pred) - tp_pred, "latencies": latencies, "days": len(y) / samples_per_day}


def summarize_counts(counts: list[dict]) -> dict:
    """Pool event counts over series: micro precision/recall/F1, false alerts per day, median latency."""
    n_gt, det = sum(c["n_gt"] for c in counts), sum(c["detected"] for c in counts)
    n_pred, tp = sum(c["n_pred"] for c in counts), sum(c["tp_pred"] for c in counts)
    days = sum(c["days"] for c in counts)
    lat = [v for c in counts for v in c["latencies"]]
    p, r = (tp / n_pred if n_pred else 0.0), (det / n_gt if n_gt else 0.0)
    return {"events_true": n_gt, "events_detected": det, "incidents_predicted": n_pred,
            "event_precision": round(p, 3), "event_recall": round(r, 3),
            "event_f1": round(2 * p * r / (p + r), 3) if p + r else 0.0,
            "false_alerts_per_day": round((n_pred - tp) / max(days, 1e-9), 3),
            "median_detection_latency_min": float(np.median(lat)) if lat else None}