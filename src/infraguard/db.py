"""SQLAlchemy models. PostgreSQL in Docker; SQLite by default for local runs."""
from __future__ import annotations

from sqlalchemy import JSON, Column, DateTime, Float, Integer, String, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from .config import BASE_METRICS, DATABASE_URL

Base = declarative_base()
_sqlite = DATABASE_URL.startswith("sqlite")
engine = create_engine(DATABASE_URL, pool_pre_ping=True,
                       connect_args={"check_same_thread": False} if _sqlite else {})
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


class Telemetry(Base):
    __tablename__ = "telemetry"
    server_id = Column(String(32), primary_key=True)
    timestamp = Column(DateTime, primary_key=True, index=True)
    for _m in BASE_METRICS:
        locals()[_m] = Column(Float, nullable=False, default=0.0)
    del _m


class Incident(Base):
    __tablename__ = "incidents"
    id = Column(Integer, primary_key=True, autoincrement=True)
    server_id = Column(String(32), index=True)
    start = Column(DateTime, index=True)
    end = Column(DateTime)
    detected_at = Column(DateTime)
    severity = Column(String(8))
    status = Column(String(8))  # open | closed
    model = Column(String(32))
    hypothesis = Column(String(200))
    recommended_action = Column(String(500))
    anomalous_metrics = Column(JSON)
    metric_events = Column(JSON)
    peak_score = Column(Float)
    peak_ratio = Column(Float)
    duration_min = Column(Integer)


class Anomaly(Base):
    __tablename__ = "anomalies"
    server_id = Column(String(32), primary_key=True)
    timestamp = Column(DateTime, primary_key=True, index=True)
    model = Column(String(32), primary_key=True)
    score = Column(Float)
    threshold = Column(Float)
    incident_id = Column(Integer, index=True, nullable=True)


def init_db() -> None:
    Base.metadata.create_all(engine)
