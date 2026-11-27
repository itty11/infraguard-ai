"""Central configuration for InfraGuard AI (override with environment variables)."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.getenv("DATA_DIR", ROOT / "data"))
ARTIFACT_DIR = Path(os.getenv("ARTIFACT_DIR", ROOT / "artifacts"))
KB_DIR = Path(os.getenv("KB_DIR", DATA_DIR / "knowledge_base"))
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{ROOT / 'infraguard.db'}")
API_KEY = os.getenv("API_KEY", "")  # if set, clients must send X-API-Key
ACTIVE_MODEL = os.getenv("ACTIVE_MODEL", "isolation_forest")

# --- RAG / assistant -------------------------------------------------------
EMBEDDING_BACKEND = os.getenv("EMBEDDING_BACKEND", "tfidf")  # tfidf | sbert
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
LLM_MODEL = os.getenv("LLM_MODEL", "claude-sonnet-5-5")  # used only if ANTHROPIC_API_KEY is set
RAG_MIN_SCORE = float(os.getenv("RAG_MIN_SCORE", "0.08"))  # below this => "insufficient documentation"

# --- Telemetry schema (one schema used everywhere) --------------------------
SERVERS = [f"SERVER_{i:02d}" for i in range(1, 5)]

BASE_METRICS = [
    "cpu_usage",          # %
    "memory_usage",       # %
    "disk_usage",         # %
    "disk_io",            # MB/s
    "network_in",         # Mbps
    "network_out",        # Mbps
    "latency",            # ms
    "packet_loss",        # %
    "connection_failures",  # count / min
    "request_rate",       # req / s
    "response_time",      # ms
    "error_count",        # count / min
    "restart_events",     # count / min
]
PERCENT_METRICS = ["cpu_usage", "memory_usage", "disk_usage", "packet_loss"]

FEATURE_COLUMNS = BASE_METRICS + [
    "cpu_mean_5", "mem_max_10", "mem_slope_10", "disk_slope_10",
    "latency_mean_5", "latency_diff_1", "latency_pct_5",
    "loss_mean_5", "resp_mean_5", "error_mean_5", "connfail_mean_5",
    "cpu_per_req", "hour_sin", "hour_cos",
]

# Columns used for the hour-of-day baseline (scoring of the statistical baseline and
# per-metric attribution of which metrics are anomalous inside an incident).
Z_COLUMNS = [
    "cpu_usage", "memory_usage", "disk_usage", "disk_io", "network_in", "network_out",
    "latency", "packet_loss", "response_time", "request_rate",
    "error_mean_5", "connfail_mean_5",
]
Z_DISPLAY = {"error_mean_5": "error_count", "connfail_mean_5": "connection_failures"}

# --- Detection settings -----------------------------------------------------
THRESHOLD_QUANTILE = 0.995   # threshold = this quantile of scores on *normal* validation data
MIN_CONSECUTIVE = 3          # an alert must persist this many samples (filters noise)
MERGE_GAP = 12               # runs separated by <= this many samples are one incident
WARMUP = 10                  # first samples per server lack rolling history -> never flagged
