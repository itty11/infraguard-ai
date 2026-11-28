import os
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
_TMP = tempfile.mkdtemp(prefix="infraguard_test_")
# set TEST_DATABASE_URL (e.g. postgresql+psycopg2://user:pw@localhost:5432/db) to run the suite on PostgreSQL
os.environ["DATABASE_URL"] = os.getenv("TEST_DATABASE_URL") or f"sqlite:///{_TMP}/test.db"
os.environ["ARTIFACT_DIR"] = f"{_TMP}/artifacts"
os.environ.pop("ANTHROPIC_API_KEY", None)
os.environ.pop("API_KEY", None)


@pytest.fixture(scope="session")
def datasets():
    from infraguard.simulator import generate_dataset
    return {
        "train": generate_dataset(3, "2026-10-01", seed=1, servers=["SERVER_01", "SERVER_02"]),
        "val": generate_dataset(1, "2026-10-04", seed=2, servers=["SERVER_01", "SERVER_02"]),
        "test": generate_dataset(2, "2026-10-05", seed=3, anomalies_per_day=6, servers=["SERVER_01", "SERVER_02"]),
    }


@pytest.fixture(scope="session")
def bundle(datasets):
    from infraguard.pipeline import DetectorBundle
    b = DetectorBundle().fit(datasets["train"], datasets["val"], ae_epochs=5)
    b.save(Path(os.environ["ARTIFACT_DIR"]))
    return b