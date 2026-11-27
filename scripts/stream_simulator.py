"""Replay simulated telemetry (with injected anomalies) into a running API, accelerated."""
import _path  # noqa: F401
import argparse
import os
import time

import pandas as pd
import requests

from infraguard.config import BASE_METRICS
from infraguard.simulator import generate_dataset

p = argparse.ArgumentParser()
p.add_argument("--api", default=os.getenv("API_URL", "http://localhost:8000"))
p.add_argument("--hours", type=float, default=6)
p.add_argument("--anomalies-per-day", type=float, default=12, help="per server (high so a short demo shows incidents)")
p.add_argument("--batch", type=int, default=5, help="timestamps per request")
p.add_argument("--delay", type=float, default=0.5, help="seconds between requests")
p.add_argument("--seed", type=int, default=7)
a = p.parse_args()

headers = {"X-API-Key": os.environ["API_KEY"]} if os.getenv("API_KEY") else {}
start = (pd.Timestamp.now(tz="UTC").tz_localize(None).floor("min") - pd.Timedelta(hours=a.hours)).isoformat()
df = generate_dataset(a.hours / 24, start=start, seed=a.seed, anomalies_per_day=a.anomalies_per_day, corrupt_rate=0)
stamps = sorted(df["timestamp"].unique())
print(f"streaming {len(stamps)} timestamps x {df.server_id.nunique()} servers to {a.api}")
for i in range(0, len(stamps), a.batch):
    chunk = df[df["timestamp"].isin(stamps[i:i + a.batch])]
    points = [{"server_id": r.server_id, "timestamp": pd.Timestamp(r.timestamp).isoformat(),
               **{m: float(getattr(r, m)) for m in BASE_METRICS}} for r in chunk.itertuples()]
    r = requests.post(f"{a.api}/telemetry", json={"points": points}, headers=headers, timeout=60)
    r.raise_for_status()
    if r.json().get("incidents_created"):
        print(f"[{pd.Timestamp(stamps[min(i + a.batch, len(stamps)) - 1])}] new incidents: {r.json()['incidents_created']}")
    time.sleep(a.delay)
print("done")
