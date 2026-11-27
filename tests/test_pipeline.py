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
