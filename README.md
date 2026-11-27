# InfraGuard AI

![CI](https://github.com/itty11/infraguard-ai/actions/workflows/ci.yml/badge.svg)

AI-powered infrastructure monitoring and anomaly detection (AIOps MVP): a telemetry simulator,
multivariate anomaly detection, incident grouping, a RAG troubleshooting assistant, a FastAPI backend and a
Streamlit dashboard. Everything runs on a laptop (CPU only).

> **Scope note.** Telemetry is *simulated*. Results below show the pipeline works and how methods compare on
> synthetic data; they are **not** measurements from a real environment. Detection is not root-cause analysis:
> every incident carries a *hypothesis*, never a confirmed cause.

## What is implemented (mapped to the plan)

| Plan layer | Where | Notes |
|---|---|---|
| 1 Telemetry generation | `src/infraguard/simulator.py` | 4 servers, 13 metrics, daily/weekly workload cycle, 8 labelled anomaly types, missing/invalid readings |
| 2 Data processing | `features.py` | cleaning without dropping rows, rolling means/max, slopes, % change, workload-normalised CPU, hour encoding |
| 3 Detection | `models.py`, `pipeline.py` | static threshold, hour-of-day z-score baseline, Isolation Forest, Autoencoder (Keras, or sklearn MLP fallback) |
| 4 Correlation | `correlation.py` | persistence filter, gap merging, per-metric attribution, sudden/gradual labels, severity rules, hypothesis |
| 5 AI assistant | `rag.py` | FAISS retrieval over `data/knowledge_base/*.md`, telemetry evidence, citations, "insufficient docs" fallback, optional LLM |
| 6 Backend | `api.py`, `db.py`, `service.py` | FastAPI + SQLAlchemy (PostgreSQL in Docker, SQLite locally) |
| 7 Dashboard | `dashboard/app.py` | overview, charts, anomaly timeline, incidents, assistant, model performance |
| Evaluation | `evaluation.py`, `scripts/evaluate.py` | event-based precision/recall/F1, false alerts/day, detection latency, grouping ARI, alert reduction |

**Not implemented (diagram extras):** SNMP collection, CMDB/impact analysis, ticketing, auto-remediation, real log
parsing (only error/connection-failure counts are used). Treat them as future work.

## Quick start (local)

```bash
python -m venv .venv && source .venv/bin/activate      # Windows PowerShell: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -r requirements-dl.txt          # optional: real Keras autoencoder (Windows: enable long paths first)
export PYTHONPATH=src                        # Windows PowerShell: $env:PYTHONPATH="src"

python scripts/generate_data.py              # train/val (normal) + test (labelled anomalies)
python scripts/train.py                      # trains on normal data, sets thresholds on normal validation data
python scripts/evaluate.py                   # compares all detectors, writes artifacts/eval_report.json

uvicorn infraguard.api:app --reload          # terminal 1  -> http://localhost:8000/docs
python scripts/stream_simulator.py           # terminal 2  -> replays telemetry with injected anomalies
streamlit run dashboard/app.py               # terminal 3  -> http://localhost:8501
pytest -q                                    # tests
```

Docker: `docker compose up --build` (add `--profile demo` for the live stream). The first start simulates data
and trains the model inside the API container.

## API

| Endpoint | Purpose |
|---|---|
| `POST /telemetry` | ingest samples (validated, idempotent) and run detection on a sliding window |
| `GET /metrics` | latest sample per server, or `?server_id=&limit=` history |
| `GET /anomalies` | flagged samples (`server_id`, `limit`) |
| `GET /incidents`, `GET /incidents/{id}` | grouped incidents (`server_id`, `severity`) |
| `POST /predict` | stateless detection on supplied samples (>= 15 per server) |
| `POST /assistant/query` | grounded troubleshooting answer with citations |
| `GET /model/report` | latest evaluation report |
| `GET /health` | DB / model / knowledge-base status |

Set `API_KEY` to require an `X-API-Key` header. Set `ANTHROPIC_API_KEY` for LLM-written answers
(`LLM_MODEL` configurable); without it the assistant returns an extractive answer built only from retrieved runbooks.

## Results from the included simulator (4 days x 4 servers, 51 injected events, default seeds)

| Model | Event precision | Event recall | False alerts / day / server | Median detection latency (min) | Grouping ARI |
|---|---|---|---|---|---|
| Static thresholds (CPU/mem >90%, latency >200 ms, loss >5%) | 1.00 | 0.26 | 0.00 | 4 | 0.03 |
| Hour-of-day z-score baseline | 0.96 | 1.00 | 0.13 | 2 | 0.67 |
| Isolation Forest | 0.90 | 0.94 | 0.38 | 2 | 0.16 |
| Autoencoder (Keras) | 0.96 | 1.00 | 0.13 | 2 | 0.52 |

Honest reading: fixed thresholds miss ~3 of 4 events; the simple workload-aware baseline is very strong on this
simulator; Isolation Forest is the weakest here and fragments some incidents (low ARI). Numbers vary with seeds
and will differ on real data. Re-run `scripts/evaluate.py` for your own figures.

## Known limitations / next steps

1. **Synthetic-data risk:** you wrote the generator and the evaluation uses it. Add a public multivariate server
   dataset (e.g. SMD) and report results there before making claims.
2. Isolation Forest sees weak temporal context; add more trend features or a windowed/LSTM autoencoder.
3. Thresholds come from a quantile of normal validation scores (`THRESHOLD_QUANTILE` in `config.py`); tune on
   validation data, never on the test set.
4. Retrieval defaults to TF-IDF vectors in FAISS (no model download). `EMBEDDING_BACKEND=sbert` uses neural embeddings.
5. Add an assistant evaluation set (question -> expected runbook) to measure retrieval relevance and groundedness.
6. Add authentication beyond a shared API key, rate limiting and model versioning before any real deployment.

## Verified vs. not verified

Run and passing on Windows (Python 3.11, pandas 3.1, scikit-learn 1.9, TensorFlow 2.21, SQLite): data generation,
training, evaluation (results above reproduced on a second machine), the 13 unit/API tests, live ingestion through
the API and the Streamlit dashboard (overview, anomaly timeline, incidents). The assistant is covered by an API test
in extractive mode. CI runs the test suite on every push (Linux, Python 3.11, without TensorFlow).
**Not run:** the Docker images / PostgreSQL path, `sbert` embeddings, and LLM mode.

Dashboard timestamps are UTC.