"""Cleaning and feature engineering (works per server, preserves sampling cadence)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import BASE_METRICS, FEATURE_COLUMNS, PERCENT_METRICS


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Sort, flag invalid readings as missing, and impute without dropping rows."""
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    for c in BASE_METRICS:
        if c not in df:
            df[c] = 0.0
    df = df.sort_values(["server_id", "timestamp"]).reset_index(drop=True)
    vals = df[BASE_METRICS].astype(float)
    vals = vals.mask(vals < 0)                                   # negative => invalid
    for c in PERCENT_METRICS:
        vals[c] = vals[c].mask(vals[c] > 100)                    # >100% => invalid
    vals["server_id"] = df["server_id"]
    vals = vals.groupby("server_id", sort=False)[BASE_METRICS].transform(
        lambda s: s.ffill(limit=5).bfill(limit=5))
    df[BASE_METRICS] = vals.fillna(vals.median()).fillna(0.0)    # last-resort fill
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add rolling / change / context features. Input must already be cleaned."""
    f = df.copy()
    g = f.groupby("server_id", sort=False)

    def roll(col, n, how="mean"):
        return g[col].transform(lambda s: getattr(s.rolling(n, min_periods=1), how)())

    def slope(col, n):
        return g[col].transform(lambda s: (s - s.shift(n)) / n).fillna(0.0)

    f["cpu_mean_5"] = roll("cpu_usage", 5)
    f["mem_max_10"] = roll("memory_usage", 10, "max")
    f["mem_slope_10"] = slope("memory_usage", 10)
    f["disk_slope_10"] = slope("disk_usage", 10)
    f["latency_mean_5"] = roll("latency", 5)
    f["latency_diff_1"] = g["latency"].diff().fillna(0.0)
    prev = g["latency"].shift(5)
    f["latency_pct_5"] = ((f["latency"] - prev) / (prev.abs() + 1e-6)).fillna(0.0).clip(-10, 10)
    f["loss_mean_5"] = roll("packet_loss", 5)
    f["resp_mean_5"] = roll("response_time", 5)
    f["error_mean_5"] = roll("error_count", 5)
    f["connfail_mean_5"] = roll("connection_failures", 5)
    f["cpu_per_req"] = f["cpu_usage"] / (f["request_rate"] + 1.0)   # workload-normalised CPU
    hour = f["timestamp"].dt.hour + f["timestamp"].dt.minute / 60.0
    f["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    f["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    assert not f[FEATURE_COLUMNS].isna().any().any(), "NaNs left after feature engineering"
    return f
