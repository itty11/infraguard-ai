"""Telemetry simulator: normal behaviour + labelled anomalies (1-minute sampling)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import lfilter

from .config import BASE_METRICS, PERCENT_METRICS, SERVERS

ANOMALY_TYPES = [
    "cpu_spike", "memory_leak", "network_congestion", "latency_surge",
    "packet_loss", "disk_saturation", "cpu_without_load", "combined",
]
DURATION = {  # minutes (min, max)
    "cpu_spike": (10, 30), "memory_leak": (60, 120), "network_congestion": (15, 40),
    "latency_surge": (10, 30), "packet_loss": (10, 30), "disk_saturation": (60, 120),
    "cpu_without_load": (20, 50), "combined": (45, 90),
}


def _ar1(n: int, phi: float, std: float, rng: np.random.Generator) -> np.ndarray:
    """AR(1) noise with a given stationary standard deviation."""
    sigma = std * np.sqrt(1 - phi ** 2)
    return lfilter([1.0], [1.0, -phi], rng.normal(0, sigma, n))


def _workload(idx: pd.DatetimeIndex) -> np.ndarray:
    """Workload factor in (0, 1]: daily cycle (peak ~15:00) with quieter weekends."""
    hour = np.asarray(idx.hour + idx.minute / 60.0, dtype=float)
    daily = 0.5 + 0.35 * np.sin(2 * np.pi * (hour - 9) / 24)
    weekly = np.where(np.asarray(idx.dayofweek) >= 5, 0.7, 1.0)
    return np.clip(daily * weekly, 0.05, 1.0)


def _baseline(start, n: int, rng: np.random.Generator, scale: float):
    idx = pd.date_range(start, periods=n, freq="1min")
    w = _workload(idx)
    t = np.arange(n)
    d = {}
    d["request_rate"] = np.maximum(5, 400 * scale * w + _ar1(n, 0.8, 12, rng))
    d["cpu_usage"] = 12 + 45 * w * scale + _ar1(n, 0.85, 2.5, rng)
    d["memory_usage"] = 35 + 12 * w + _ar1(n, 0.98, 2.0, rng)
    d["disk_usage"] = 60 + 4 * np.sin(2 * np.pi * t / (3 * 1440)) + _ar1(n, 0.99, 0.4, rng)
    d["disk_io"] = np.maximum(0, 20 + 80 * w + _ar1(n, 0.7, 5, rng))
    d["network_in"] = np.maximum(0, 0.25 * d["request_rate"] + _ar1(n, 0.8, 3, rng))
    d["network_out"] = np.maximum(0, 0.6 * d["network_in"] + _ar1(n, 0.8, 2, rng))
    d["latency"] = np.maximum(1, 18 + 6 * w + _ar1(n, 0.7, 1.5, rng))
    d["response_time"] = np.maximum(10, 90 + 50 * w + _ar1(n, 0.7, 6, rng))
    d["packet_loss"] = np.abs(_ar1(n, 0.5, 0.08, rng)) + 0.02
    d["connection_failures"] = rng.poisson(0.3, n).astype(float)
    d["error_count"] = rng.poisson(0.5 + 1.0 * w).astype(float)
    d["restart_events"] = np.zeros(n)
    return d, idx


def _inject(d: dict, kind: str, s: int, dur: int, rng: np.random.Generator) -> None:
    sl = slice(s, s + dur)
    sudden = np.minimum(1.0, np.arange(1, dur + 1) / 3.0)   # ramps in over 3 minutes
    grad = np.linspace(0.05, 1.0, dur)                      # slow ramp over the whole event
    u = rng.uniform

    def add(col, amount):
        d[col][sl] += amount

    if kind == "cpu_spike":
        add("cpu_usage", u(30, 45) * sudden)
        add("response_time", u(60, 120) * sudden)
        add("latency", u(5, 12) * sudden)
        add("error_count", rng.poisson(2.0 * sudden))
    elif kind == "cpu_without_load":  # abnormal, yet below any fixed 90% threshold
        add("cpu_usage", u(18, 28) * sudden)
        d["request_rate"][sl] *= 1 - 0.35 * sudden
        add("response_time", u(25, 50) * sudden)
    elif kind == "memory_leak":
        add("memory_usage", u(30, 45) * grad)
        add("response_time", u(40, 90) * grad ** 2)
        add("error_count", rng.poisson(1.5 * grad ** 2))
    elif kind == "network_congestion":
        d["latency"][sl] *= 1 + u(3, 6) * sudden
        add("packet_loss", u(2, 6) * sudden)
        d["network_in"][sl] *= 1 + 0.6 * sudden
        d["network_out"][sl] *= 1 + 0.6 * sudden
        add("connection_failures", rng.poisson(2.0 * sudden))
        add("response_time", u(50, 120) * sudden)
    elif kind == "latency_surge":
        bump = u(80, 200) * sudden
        add("latency", bump)
        add("response_time", 0.6 * bump)
    elif kind == "packet_loss":
        add("packet_loss", u(3, 10) * sudden)
        add("connection_failures", rng.poisson(3.0 * sudden))
        add("latency", u(10, 30) * sudden)
    elif kind == "disk_saturation":
        add("disk_usage", u(18, 28) * grad)
        add("disk_io", u(80, 140) * grad)
        add("response_time", u(30, 60) * grad)
        add("error_count", rng.poisson(1.0 * grad))
    elif kind == "combined":  # CPU + memory + latency + packet loss together (INC example in the plan)
        add("cpu_usage", u(30, 45) * sudden)
        add("memory_usage", u(20, 30) * grad)
        add("latency", u(60, 150) * sudden)
        add("packet_loss", u(3, 8) * sudden)
        add("response_time", u(80, 140) * sudden)
        add("error_count", rng.poisson(3.0 * sudden))
        d["restart_events"][s + dur - 1] = 1.0
    else:
        raise ValueError(kind)


def _schedule(n: int, rng: np.random.Generator, per_day: float, min_gap: int = 45):
    events, occupied = [], np.zeros(n, dtype=bool)
    for _ in range(rng.poisson(per_day * n / 1440)):
        for _try in range(25):
            kind = str(rng.choice(ANOMALY_TYPES))
            lo, hi = DURATION[kind]
            dur = int(rng.integers(lo, hi + 1))
            if n < dur + 60:
                continue
            s = int(rng.integers(30, n - dur - 5))
            a, b = max(0, s - min_gap), min(n, s + dur + min_gap)
            if not occupied[a:b].any():
                occupied[a:b] = True
                events.append((kind, s, dur))
                break
    return sorted(events, key=lambda e: e[1])


def _clip(d: dict) -> None:
    for c in PERCENT_METRICS:
        d[c] = np.clip(d[c], 0, 100)
    for c in BASE_METRICS:
        d[c] = np.maximum(d[c], 0)


def _corrupt(df: pd.DataFrame, rng: np.random.Generator, rate: float) -> pd.DataFrame:
    """Inject missing and invalid (negative) readings, like a flaky collector."""
    vals = df[BASE_METRICS].to_numpy(dtype=float, copy=True)
    r = rng.random(vals.shape)
    vals[r < rate] = np.nan
    vals[(r >= rate) & (r < rate * 1.25)] = -1.0
    df[BASE_METRICS] = vals
    return df


def generate_dataset(days: float, start: str = "2026-10-01", seed: int = 0,
                     anomalies_per_day: float = 0.0, servers: list[str] | None = None,
                     corrupt_rate: float = 0.002) -> pd.DataFrame:
    """Return raw telemetry for all servers plus ground-truth labels.

    Label columns: is_anomaly (0/1), anomaly_type, true_incident_id (0 = normal).
    """
    rng = np.random.default_rng(seed)
    servers = servers or SERVERS
    n = int(days * 1440)
    frames, counter = [], 0
    for sid in servers:
        d, idx = _baseline(start, n, rng, scale=float(rng.uniform(0.9, 1.1)))
        is_anom = np.zeros(n, dtype=int)
        a_type = np.full(n, "", dtype=object)
        inc_id = np.zeros(n, dtype=int)
        for kind, s, dur in _schedule(n, rng, anomalies_per_day):
            counter += 1
            _inject(d, kind, s, dur, rng)
            is_anom[s:s + dur], a_type[s:s + dur], inc_id[s:s + dur] = 1, kind, counter
        _clip(d)
        df = pd.DataFrame(d)
        df.insert(0, "timestamp", idx)
        df.insert(1, "server_id", sid)
        df["is_anomaly"], df["anomaly_type"], df["true_incident_id"] = is_anom, a_type, inc_id
        frames.append(df)
    out = pd.concat(frames).sort_values(["timestamp", "server_id"]).reset_index(drop=True)
    return _corrupt(out, rng, corrupt_rate) if corrupt_rate else out
