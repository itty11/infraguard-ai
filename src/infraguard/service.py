"""Ingest telemetry, run detection on a sliding window, and persist anomalies/incidents."""
from __future__ import annotations

from datetime import timedelta

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from .config import BASE_METRICS, MERGE_GAP, WARMUP
from .correlation import ACTIONS, JOINT_HYPOTHESIS, _hypothesis
from .db import Anomaly, Incident, Telemetry
from .pipeline import DetectorBundle, detect

SEVERITY_RANK = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
WINDOW = 240        # samples of history re-scored on every ingest
MIN_POINTS = WARMUP + 5


def _fetch_window(session: Session, server_id: str, limit: int = WINDOW) -> pd.DataFrame:
    rows = session.execute(select(Telemetry).where(Telemetry.server_id == server_id)
                           .order_by(Telemetry.timestamp.desc()).limit(limit)).scalars().all()
    cols = ["server_id", "timestamp", *BASE_METRICS]
    return pd.DataFrame([{c: getattr(r, c) for c in cols} for r in reversed(rows)], columns=cols)


def ingest(session: Session, bundle: DetectorBundle | None, points: list[dict]) -> dict:
    for p in points:  # merge() = idempotent upsert on (server_id, timestamp)
        session.merge(Telemetry(**{k: p[k] for k in ["server_id", "timestamp", *BASE_METRICS]}))
    session.commit()
    new_incidents = 0
    if bundle is not None:
        for sid in sorted({p["server_id"] for p in points}):
            new_incidents += process_server(session, bundle, sid)
    return {"ingested": len(points), "incidents_created": new_incidents}


def process_server(session: Session, bundle: DetectorBundle, server_id: str) -> int:
    df = _fetch_window(session, server_id)
    if len(df) < MIN_POINTS:
        return 0
    feats, _, incidents = detect(bundle, df)
    last_ts = feats["timestamp"].max()
    created = 0
    for inc in incidents:
        start, end = inc["start"].to_pydatetime(), inc["end"].to_pydatetime()
        gap = timedelta(minutes=MERGE_GAP + 1)
        row = session.execute(select(Incident).where(
            Incident.server_id == server_id, Incident.start <= end + gap, Incident.end >= start - gap)
            .order_by(Incident.start)).scalars().first()
        if row is None:
            row, created = Incident(server_id=server_id), created + 1
            session.add(row)
        row.start = min(row.start, start) if row.start else start
        row.end = max(row.end, end) if row.end else end
        row.detected_at = row.detected_at or inc["detected_at"].to_pydatetime()
        # The detection window slides, so one evaluation may see only part of a long incident.
        # Merge instead of overwrite: severity never drops, metrics are only ever added.
        row.model = inc["model"]
        if row.severity is None or SEVERITY_RANK[inc["severity"]] >= SEVERITY_RANK[row.severity]:
            row.severity = inc["severity"]
        metrics = list(dict.fromkeys([*(row.anomalous_metrics or []), *inc["anomalous_metrics"]]))
        row.anomalous_metrics = metrics                                   # new objects so SQLAlchemy sees the change
        row.metric_events = {**inc["metric_events"], **(row.metric_events or {})}
        if inc["hypothesis"] == JOINT_HYPOTHESIS:
            # weak evidence: keep a firmer hypothesis if an earlier evaluation already produced one
            row.hypothesis = row.hypothesis if row.hypothesis not in (None, JOINT_HYPOTHESIS) else JOINT_HYPOTHESIS
        else:
            row.hypothesis = _hypothesis(metrics, row.metric_events)
        row.recommended_action = ACTIONS.get(row.hypothesis, ACTIONS["Unclassified anomaly"])
        row.peak_score = max(row.peak_score or 0.0, inc["peak_score"])
        row.peak_ratio = max(row.peak_ratio or 0.0, inc["peak_ratio"])
        row.duration_min = int((row.end - row.start).total_seconds() // 60) + 1
        row.status = "open" if (last_ts - pd.Timestamp(row.end)) <= pd.Timedelta(minutes=MERGE_GAP) else "closed"
        session.flush()
        flagged = feats[(feats["flag"]) & feats["timestamp"].between(inc["start"], inc["end"])]
        for r in flagged.itertuples():
            session.merge(Anomaly(server_id=server_id, timestamp=r.timestamp.to_pydatetime(), model=inc["model"],
                                  score=float(r.score), threshold=float(r.threshold), incident_id=row.id))
    session.commit()
    return created