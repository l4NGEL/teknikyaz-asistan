"""API isteklerinin izlenebilirlik metriklerini tutan hafif bir SQLite deposu.

MLflow eğitim/deney metrikleri için uygundur ama canlı istek trafiğini (üretim izleme)
loglamak için ağırdır. Burada FastAPI'nin her isteği yazdığı, Streamlit dashboard'unun
okuduğu basit, bağımlılıksız bir SQLite tablosu kullanıyoruz — küçük ölçekli LLMOps
izleme için gerçekçi ve yeterli bir çözüm.
"""
from __future__ import annotations

import sqlite3
import time
from contextlib import contextmanager

from src.config import METRICS_DB_PATH

_SCHEMA = """
CREATE TABLE IF NOT EXISTS requests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp REAL NOT NULL,
    endpoint TEXT NOT NULL,
    model_version TEXT,
    latency_ms REAL NOT NULL,
    hallucination_score REAL,
    input_preview TEXT,
    output_preview TEXT
);
"""


@contextmanager
def _connect():
    conn = sqlite3.connect(METRICS_DB_PATH)
    try:
        conn.execute(_SCHEMA)
        yield conn
        conn.commit()
    finally:
        conn.close()


def log_request(
    endpoint: str,
    latency_ms: float,
    model_version: str | None = None,
    hallucination_score: float | None = None,
    input_preview: str = "",
    output_preview: str = "",
) -> None:
    with _connect() as conn:
        conn.execute(
            "INSERT INTO requests (timestamp, endpoint, model_version, latency_ms, "
            "hallucination_score, input_preview, output_preview) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                time.time(), endpoint, model_version, latency_ms, hallucination_score,
                input_preview[:200], output_preview[:200],
            ),
        )


def fetch_recent(limit: int = 200) -> list[dict]:
    with _connect() as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM requests ORDER BY timestamp DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]
