"""FastAPI service: ingest telemetry, serve metrics/anomalies/incidents, run the assistant."""
from __future__ import annotations

import json
from contextlib import asynccontextmanager
from datetime import datetime

import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import select, text
from sqlalchemy.orm import Session

from . import rag, service
from .config import ACTIVE_MODEL, API_KEY, ARTIFACT_DIR, BASE_METRICS
from .db import Anomaly, Incident, SessionLocal, Telemetry, init_db
from .pipeline import DetectorBundle, detect

STATE: dict = {"bundle": None, "kb": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    if (ARTIFACT_DIR / "bundle.joblib").exists():
        STATE["bundle"] = DetectorBundle.load(ARTIFACT_DIR)
    try:
        STATE["kb"] = rag.KnowledgeBase()
    except Exception as exc:  # keep the API up even if the KB is missing
        print("knowledge base unavailable:", exc)
    yield


app = FastAPI(title="InfraGuard AI", version="0.1.0", lifespan=lifespan)


def require_key(x_api_key: str | None = Header(default=None)):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(401, "invalid or missing X-API-Key")


def get_session():
    with SessionLocal() as s:
        yield s


# ------------------------------------------------------------------ schemas
class TelemetryPoint(BaseModel):
    server_id: str = Field(min_length=1, max_length=32)
    timestamp: datetime
    cpu_usage: float = Field(ge=0, le=100)
    memory_usage: float = Field(ge=0, le=100)
    disk_usage: float = Field(ge=0, le=100)
    disk_io: float = Field(ge=0)
    network_in: float = Field(ge=0)
    network_out: float = Field(ge=0)
    latency: float = Field(ge=0)
    packet_loss: float = Field(ge=0, le=100)
    connection_failures: float = Field(default=0, ge=0)
    request_rate: float = Field(ge=0)
    response_time: float = Field(ge=0)
    error_count: float = Field(default=0, ge=0)
    restart_events: float = Field(default=0, ge=0)


class TelemetryBatch(BaseModel):
    points: list[TelemetryPoint] = Field(min_length=1, max_length=5000)


class AssistantQuery(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    server_id: str | None = None
    incident_id: int | None = None


def _incident_dict(i: Incident) -> dict:
    return {"incident_id": f"INC-{i.id:03d}", "id": i.id, "server_id": i.server_id, "severity": i.severity,
            "status": i.status, "start": i.start.isoformat(), "end": i.end.isoformat(),
            "duration_min": i.duration_min, "anomalous_metrics": i.anomalous_metrics,
            "metric_events": i.metric_events, "hypothesis": i.hypothesis,
            "recommended_action": i.recommended_action, "model": i.model,
            "note": "Hypothesis only - detection does not prove root cause."}


# ------------------------------------------------------------------ endpoints
@app.post("/telemetry", dependencies=[Depends(require_key)])
def post_telemetry(batch: TelemetryBatch, session: Session = Depends(get_session)):
    result = service.ingest(session, STATE["bundle"], [p.model_dump() for p in batch.points])
    result["detection_active"] = STATE["bundle"] is not None
    return result


@app.get("/metrics", dependencies=[Depends(require_key)])
def get_metrics(server_id: str | None = None, limit: int = Query(120, ge=1, le=5000),
                session: Session = Depends(get_session)):
    """With server_id: last `limit` samples (oldest first). Without: latest sample of every server."""
    if server_id:
        rows = session.execute(select(Telemetry).where(Telemetry.server_id == server_id)
                               .order_by(Telemetry.timestamp.desc()).limit(limit)).scalars().all()[::-1]
    else:
        ids = session.execute(select(Telemetry.server_id).distinct()).scalars().all()
        rows = [session.execute(select(Telemetry).where(Telemetry.server_id == s)
                                .order_by(Telemetry.timestamp.desc()).limit(1)).scalars().first() for s in sorted(ids)]
    return [{"server_id": r.server_id, "timestamp": r.timestamp.isoformat(),
             **{m: getattr(r, m) for m in BASE_METRICS}} for r in rows]


@app.get("/anomalies", dependencies=[Depends(require_key)])
def get_anomalies(server_id: str | None = None, limit: int = Query(500, ge=1, le=5000),
                  session: Session = Depends(get_session)):
    q = select(Anomaly).order_by(Anomaly.timestamp.desc()).limit(limit)
    if server_id:
        q = select(Anomaly).where(Anomaly.server_id == server_id).order_by(Anomaly.timestamp.desc()).limit(limit)
    return [{"server_id": a.server_id, "timestamp": a.timestamp.isoformat(), "score": a.score,
             "threshold": a.threshold, "incident_id": a.incident_id, "model": a.model}
            for a in session.execute(q).scalars().all()]


@app.get("/incidents", dependencies=[Depends(require_key)])
def get_incidents(server_id: str | None = None, severity: str | None = None,
                  limit: int = Query(100, ge=1, le=1000), session: Session = Depends(get_session)):
    q = select(Incident).order_by(Incident.start.desc()).limit(limit)
    if server_id:
        q = q.where(Incident.server_id == server_id)
    if severity:
        q = q.where(Incident.severity == severity.upper())
    return [_incident_dict(i) for i in session.execute(q).scalars().all()]


@app.get("/incidents/{incident_id}", dependencies=[Depends(require_key)])
def get_incident(incident_id: int, session: Session = Depends(get_session)):
    inc = session.get(Incident, incident_id)
    if inc is None:
        raise HTTPException(404, "incident not found")
    return _incident_dict(inc)


@app.post("/predict", dependencies=[Depends(require_key)])
def predict(batch: TelemetryBatch):
    """Stateless detection on the supplied points (send >=15 consecutive samples per server)."""
    bundle = STATE["bundle"]
    if bundle is None:
        raise HTTPException(503, "no trained model - run scripts/train.py")
    df = pd.DataFrame([p.model_dump() for p in batch.points])
    if df.groupby("server_id").size().min() < service.MIN_POINTS:
        raise HTTPException(422, f"need at least {service.MIN_POINTS} samples per server")
    feats, _, incidents = detect(bundle, df)
    records = []
    for n, inc in enumerate(incidents, 1):
        records.append({"server_id": inc["server_id"], "timestamp": inc["start"].isoformat(),
                        "anomaly_detected": True, "severity": inc["severity"],
                        "anomalous_metrics": inc["anomalous_metrics"], "incident_id": f"INC-{n:03d}",
                        "anomaly_score": round(inc["peak_score"], 4), "model": inc["model"],
                        "window_start": inc["start"].isoformat(), "window_end": inc["end"].isoformat(),
                        "recommended_action": inc["recommended_action"]})
    return {"anomaly_detected": bool(records), "records": records,
            "points_flagged": int(feats["flag"].sum()), "points_scored": len(feats)}


@app.post("/assistant/query", dependencies=[Depends(require_key)])
def assistant_query(q: AssistantQuery, session: Session = Depends(get_session)):
    if STATE["kb"] is None:
        raise HTTPException(503, "knowledge base not loaded")
    return rag.answer(session, STATE["kb"], q.question, q.server_id, q.incident_id)


@app.get("/model/report")
def model_report():
    path = ARTIFACT_DIR / "eval_report.json"
    if not path.exists():
        raise HTTPException(404, "run scripts/evaluate.py first")
    return json.loads(path.read_text())


@app.get("/health")
def health(session: Session = Depends(get_session)):
    try:
        session.execute(text("SELECT 1"))
        db_ok = True
    except Exception:
        db_ok = False
    ok = db_ok and STATE["bundle"] is not None
    return {"status": "ok" if ok else "degraded", "database": db_ok,
            "model_loaded": STATE["bundle"] is not None, "active_model": ACTIVE_MODEL,
            "knowledge_base_chunks": len(STATE["kb"].chunks) if STATE["kb"] else 0}
