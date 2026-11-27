import pandas as pd
import pytest
from fastapi.testclient import TestClient

from infraguard.config import BASE_METRICS


@pytest.fixture(scope="module")
def client(bundle):
    from infraguard.api import app
    with TestClient(app) as c:
        yield c


def _points(df):
    return [{"server_id": r.server_id, "timestamp": pd.Timestamp(r.timestamp).isoformat(),
             **{m: float(getattr(r, m)) for m in BASE_METRICS}} for r in df.itertuples()]


def test_health(client):
    h = client.get("/health").json()
    assert h["status"] == "ok" and h["model_loaded"] and h["knowledge_base_chunks"] > 0


def test_validation_rejects_bad_payload(client):
    bad = {"points": [{"server_id": "S", "timestamp": "2026-10-01T00:00:00", "cpu_usage": 150}]}
    assert client.post("/telemetry", json=bad).status_code == 422


def test_ingest_incidents_and_assistant(client, datasets):
    test = datasets["test"][datasets["test"]["server_id"] == "SERVER_01"].copy()
    clean_rows = test.assign(**{m: test[m].fillna(0).clip(lower=0) for m in BASE_METRICS})
    total = 0
    for i in range(0, len(clean_rows), 300):
        r = client.post("/telemetry", json={"points": _points(clean_rows.iloc[i:i + 300])})
        assert r.status_code == 200
        total += r.json()["ingested"]
    assert total == len(clean_rows)
    assert client.post("/telemetry", json={"points": _points(clean_rows.iloc[:5])}).status_code == 200  # idempotent
    inc = client.get("/incidents", params={"server_id": "SERVER_01"}).json()
    assert inc and {"severity", "anomalous_metrics", "recommended_action"} <= inc[0].keys()
    assert client.get("/anomalies").json()
    assert len(client.get("/metrics", params={"server_id": "SERVER_01", "limit": 50}).json()) == 50
    ans = client.post("/assistant/query", json={"question": "Why is Server 01 having problems?"}).json()
    assert ans["incident_id"] and ans["sources"] and "not a confirmed root cause" in ans["answer"]


def test_predict_is_stateless(client, datasets):
    df = datasets["test"][datasets["test"]["server_id"] == "SERVER_02"].iloc[:600].copy()
    df = df.assign(**{m: df[m].fillna(0).clip(lower=0) for m in BASE_METRICS})
    r = client.post("/predict", json={"points": _points(df)})
    assert r.status_code == 200 and r.json()["points_scored"] == 600
    assert client.post("/predict", json={"points": _points(df.iloc[:3])}).status_code == 422


def test_assistant_without_incidents_is_honest(client):
    ans = client.post("/assistant/query", json={"question": "Why is Server 99 slow?", "incident_id": 99999}).json()
    assert ans["incident_id"] is None and ans["sources"] == []


def test_incident_updates_never_downgrade_or_drop_evidence(client, datasets):
    """As the sliding window moves, an ongoing incident may grow but must not lose severity or metrics."""
    rank = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
    df = datasets["test"][datasets["test"]["server_id"] == "SERVER_02"].copy()
    df = df.assign(**{m: df[m].fillna(0).clip(lower=0) for m in BASE_METRICS})
    seen = {}
    for i in range(0, len(df), 40):
        assert client.post("/telemetry", json={"points": _points(df.iloc[i:i + 40])}).status_code == 200
        for inc in client.get("/incidents", params={"server_id": "SERVER_02"}).json():
            prev = seen.get(inc["id"])
            if prev:
                assert rank[inc["severity"]] >= rank[prev["severity"]], f"{inc['incident_id']} severity downgraded"
                assert set(prev["anomalous_metrics"]) <= set(inc["anomalous_metrics"]), \
                    f"{inc['incident_id']} lost metrics"
            seen[inc["id"]] = inc
    assert seen, "expected incidents on SERVER_02"