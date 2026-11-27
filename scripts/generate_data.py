"""Generate train (normal), validation (normal) and test (with labelled anomalies) datasets."""
import _path  # noqa: F401
import argparse

from infraguard.config import DATA_DIR
from infraguard.simulator import generate_dataset

p = argparse.ArgumentParser()
p.add_argument("--train-days", type=float, default=7)
p.add_argument("--val-days", type=float, default=2)
p.add_argument("--test-days", type=float, default=4)
p.add_argument("--anomalies-per-day", type=float, default=3.0, help="per server, test set only")
a = p.parse_args()

DATA_DIR.mkdir(parents=True, exist_ok=True)
sets = {
    "train": generate_dataset(a.train_days, "2026-10-01", seed=1),
    "val": generate_dataset(a.val_days, "2026-10-08", seed=2),
    "test": generate_dataset(a.test_days, "2026-10-10", seed=3, anomalies_per_day=a.anomalies_per_day),
}
for name, df in sets.items():
    df.to_csv(DATA_DIR / f"{name}.csv", index=False)
    print(f"{name}: {len(df):,} rows, {int(df.true_incident_id.nunique()) - 1 if name == 'test' else 0} incidents")
