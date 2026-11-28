"""Group per-sample anomaly flags into incidents with severity and a *hypothesis*.

Detection is not root-cause analysis: the hypothesis text is a hint for engineers,
never a confirmed cause.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import MERGE_GAP, MIN_CONSECUTIVE, Z_COLUMNS, Z_DISPLAY

# Resource categories drive the hypothesis; response_time / error_count / request_rate are
# application-level *symptoms* that move with almost every problem, so they do not.
CATEGORIES = {
    "compute": {"cpu_usage"},
    "memory": {"memory_usage"},
    "network": {"latency", "packet_loss", "network_in", "network_out", "connection_failures"},
    "storage": {"disk_usage", "disk_io"},
}
JOINT_HYPOTHESIS = "Unusual combination of metrics (no single metric clearly abnormal) - low confidence"
ACTIONS = {
    "Infrastructure performance degradation":
        "Review recent workload/deployment changes, then inspect server resource usage and network congestion or interface errors",
    "Possible memory leak or growing memory consumption":
        "Inspect per-process memory growth and recent releases; check for leaks and restart history",
    "Resource pressure with network impact":
        "Check CPU saturation first, then network congestion, retransmissions and interface errors",
    "Compute saturation or CPU load not explained by workload":
        "Identify top CPU processes and compare CPU with request rate; look for runaway jobs",
    "Network congestion or connectivity degradation":
        "Review link utilisation, interface errors/drops, and upstream device health",
    "Storage pressure":
        "Check disk growth, log rotation and I/O wait; identify the largest writers",
    "Application performance degradation without a clear resource cause":
        "Check recent deployments and slow dependencies (database, downstream APIs); compare with request rate",
    JOINT_HYPOTHESIS:
        "Treat as low confidence: compare the listed metrics with their normal range, check whether it persists, and "
        "follow the general triage runbook before acting",
    "Unclassified anomaly":
        "Review the affected metrics against recent changes and consult the general triage runbook",
}


def find_runs(flags: np.ndarray, min_len: int = MIN_CONSECUTIVE, max_gap: int = MERGE_GAP):
    """Runs (start, end inclusive) of True lasting >= min_len, merged when gaps <= max_gap."""
    runs, i, n = [], 0, len(flags)
    while i < n:
        if flags[i]:
            j = i
            while j + 1 < n and flags[j + 1]:
                j += 1
            if j - i + 1 >= min_len:
                runs.append([i, j])
            i = j + 1
        else:
            i += 1
    merged = []
    for r in runs:
        if merged and r[0] - merged[-1][1] - 1 <= max_gap:
            merged[-1][1] = r[1]
        else:
            merged.append(r)
    return [(a, b) for a, b in merged]


def _shape(signed: np.ndarray) -> str:
    """'gradual' if the smoothed signal needs >= 8 samples to go from onset (z>=2) to half its peak."""
    sm = pd.Series(signed).rolling(3, min_periods=1).mean().to_numpy()
    peak = sm.max()
    if peak < 2:
        return "sudden"
    ok = sm >= 2
    start = next((i for i in range(len(ok) - 2) if ok[i:i + 3].all()), None)
    if start is None:
        return "sudden"
    half = int(np.argmax(sm >= 0.5 * peak))
    return "gradual" if half - start >= 8 else "sudden"


def _severity(n_metrics: int, peak_z: float, duration: int) -> str:
    """Rule-based: how many metrics, how far from normal (max |z|), how long. HIGH>=5, MEDIUM>=3."""
    pts = (2 if n_metrics >= 4 else 1 if n_metrics >= 2 else 0)
    pts += (2 if peak_z >= 12 else 1 if peak_z >= 6 else 0) + (1 if duration >= 15 else 0)
    return "HIGH" if pts >= 5 else "MEDIUM" if pts >= 3 else "LOW"


def _hypothesis(metrics: list[str], events: dict[str, str]) -> str:
    cats = {c for c, members in CATEGORIES.items() if members & set(metrics)}
    if len(cats) >= 3:
        return "Infrastructure performance degradation"
    if cats == {"memory"}:
        return "Possible memory leak or growing memory consumption"
    if {"compute", "network"} <= cats:
        return "Resource pressure with network impact"
    if "compute" in cats:
        return "Compute saturation or CPU load not explained by workload"
    if "network" in cats:
        return "Network congestion or connectivity degradation"
    if "storage" in cats:
        return "Storage pressure"
    if {"response_time", "error_count", "request_rate"} & set(metrics):
        return "Application performance degradation without a clear resource cause"
    return "Unclassified anomaly"


def correlate(feats: pd.DataFrame, z: pd.DataFrame, model: str,
              min_consecutive: int = MIN_CONSECUTIVE, merge_gap: int = MERGE_GAP) -> list[dict]:
    """feats needs columns server_id, timestamp, score, threshold, flag; z aligns with feats.index."""
    incidents = []
    for sid, g in feats.groupby("server_id", sort=False):
        flags = g["flag"].to_numpy()
        zz = z.loc[g.index].to_numpy()
        ts, sc = g["timestamp"].tolist(), g["score"].to_numpy()
        thr = float(g["threshold"].iloc[0])
        for s, e in find_runs(flags, min_consecutive, merge_gap):
            seg = np.abs(zz[s:e + 1])
            mean_abs, max_abs = seg.mean(0), seg.max(0)
            idx = [j for j in range(len(Z_COLUMNS)) if mean_abs[j] >= 3 or max_abs[j] >= 6]
            joint = not idx
            if joint:  # flagged by the joint pattern rather than any single metric
                idx = list(np.argsort(max_abs)[::-1][:2])
            idx = sorted(idx, key=lambda j: -max_abs[j])
            lo = max(0, s - 60)
            names, events = [], {}
            for j in idx:
                name = Z_DISPLAY.get(Z_COLUMNS[j], Z_COLUMNS[j])
                sign = 1.0 if zz[s:e + 1, j].mean() >= 0 else -1.0
                shape = _shape(sign * zz[lo:e + 1, j])
                names.append(name)
                events[name] = f"{shape.capitalize()} {'increase' if sign > 0 else 'decrease'}"
            peak_ratio = float(sc[s:e + 1].max() / thr) if thr > 0 else 1.0
            duration = e - s + 1
            hyp = JOINT_HYPOTHESIS if joint else _hypothesis(names, events)
            incidents.append({
                "server_id": sid, "model": model,
                "start": ts[s], "end": ts[e], "detected_at": ts[min(s + min_consecutive - 1, e)],
                "duration_min": duration,
                # joint-pattern-only detections are low confidence by definition: never above LOW
                "severity": "LOW" if joint else _severity(len(names), float(max_abs[idx].max()), duration),
                "peak_abs_z": round(float(max_abs[idx].max()), 2),
                "anomalous_metrics": names, "metric_events": events,
                "peak_score": float(sc[s:e + 1].max()), "peak_ratio": round(peak_ratio, 3),
                "joint_pattern_only": joint,
                "hypothesis": hyp, "recommended_action": ACTIONS[hyp],
            })
    incidents.sort(key=lambda i: (i["start"], i["server_id"]))
    return incidents