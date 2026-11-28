import numpy as np
import pandas as pd

from infraguard.config import BASE_METRICS, FEATURE_COLUMNS
from infraguard.correlation import find_runs
from infraguard.evaluation import evaluate_run
from infraguard.features import build_features, clean
from infraguard.pipeline import detect
from infraguard.rag import KnowledgeBase


def test_simulator_labels_and_schema(datasets):
    t = datasets["test"]
    assert set(BASE_METRICS) <= set(t.columns)
    assert t["is_anomaly"].sum() > 0 and t["true_incident_id"].max() > 0
    assert datasets["train"]["is_anomaly"].sum() == 0          # training data is normal-only
    assert t.isna().any().any()                                # collector faults were injected


def test_clean_and_features_have_no_nans(datasets):
    f = build_features(clean(datasets["test"]))
    assert not f[FEATURE_COLUMNS].isna().any().any()
    assert (f[BASE_METRICS] >= 0).all().all()


def test_find_runs_filters_noise_and_merges_gaps():
    flags = np.array([0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0], dtype=bool)
    assert find_runs(flags, min_len=3, max_gap=2) == [(4, 11)]
    assert find_runs(flags, min_len=3, max_gap=0) == [(4, 6), (9, 11)]


def test_detection_beats_static_thresholds(bundle, datasets):
    res = {}
    for name in ("static_threshold", "zscore_baseline", "isolation_forest"):
        feats, _, inc = detect(bundle, datasets["test"], model=name)
        res[name] = evaluate_run(feats, inc)
    assert res["isolation_forest"]["event_recall"] >= 0.7
    assert res["isolation_forest"]["event_recall"] > res["static_threshold"]["event_recall"]
    assert res["isolation_forest"]["false_alerts_per_day_per_server"] < 2


def test_incident_record_fields(bundle, datasets):
    _, _, inc = detect(bundle, datasets["test"])
    assert inc, "expected at least one incident"
    for i in inc:
        assert i["severity"] in {"LOW", "MEDIUM", "HIGH"} and i["anomalous_metrics"] and i["hypothesis"]


def test_bundle_roundtrip(bundle, tmp_path):
    from infraguard.pipeline import DetectorBundle
    bundle.save(tmp_path)
    again = DetectorBundle.load(tmp_path)
    assert again.thresholds == bundle.thresholds


def test_retrieval_finds_right_runbook():
    kb = KnowledgeBase()
    assert kb.search("memory usage keeps growing over hours", 1)[0]["doc"] == "memory_leak_runbook.md"
    assert kb.search("packet loss and connection failures", 1)[0]["doc"] == "packet_loss_runbook.md"


def test_event_counts_and_summary():
    from infraguard.evaluation import event_counts, summarize_counts
    y = np.zeros(200, dtype=int)
    y[20:40] = 1        # event A: detected
    y[100:110] = 1      # event B: missed
    flags = np.zeros(200, dtype=bool)
    flags[22:30] = True     # overlaps A, first alert at 22 + 3 - 1 = 24 -> latency 4
    flags[150:160] = True   # false alert
    flags[170] = True       # single-sample blip, filtered by persistence
    c = event_counts(y, flags)
    assert (c["n_gt"], c["detected"], c["n_pred"], c["fp"]) == (2, 1, 2, 1)
    assert c["latencies"] == [4]
    s = summarize_counts([c])
    assert s["event_recall"] == 0.5 and s["event_precision"] == 0.5


def test_joint_pattern_incident_is_low_confidence():
    """Flagged by the model but no single metric clearly abnormal -> LOW severity, honest hypothesis."""
    from infraguard.config import Z_COLUMNS
    from infraguard.correlation import JOINT_HYPOTHESIS, correlate
    n = 60
    feats = pd.DataFrame({"server_id": "SERVER_01", "timestamp": pd.date_range("2026-10-01", periods=n, freq="1min"),
                          "score": 0.1, "threshold": 0.5, "flag": False})
    feats.loc[20:30, ["score", "flag"]] = [0.9, True]
    z = pd.DataFrame(np.random.default_rng(0).normal(0, 0.5, (n, len(Z_COLUMNS))), columns=Z_COLUMNS)  # all small
    (inc,) = correlate(feats, z, "isolation_forest")
    assert inc["joint_pattern_only"] and inc["severity"] == "LOW" and inc["hypothesis"] == JOINT_HYPOTHESIS


def test_assistant_warns_when_evidence_is_weak():
    from types import SimpleNamespace
    from infraguard.rag import _extractive_answer
    inc = SimpleNamespace(id=12, server_id="SERVER_02", severity="MEDIUM", start=pd.Timestamp("2026-11-27 04:48"),
                          end=pd.Timestamp("2026-11-27 04:58"), anomalous_metrics=["latency"],
                          hypothesis="Network congestion or connectivity degradation")
    hits = [{"id": "KB19", "section": "Investigation steps", "text": "- Check interface errors"}]
    weak = _extractive_answer(inc, ["latency: 18.7 -> 19.8 (+6%)"], hits, max_change=0.06)
    strong = _extractive_answer(inc, ["latency: 20 -> 80 (+300%)"], hits, max_change=3.0)
    assert "weak evidence" in weak and "6%" in weak
    assert "weak evidence" not in strong