from src import config
from src.llmops import metrics_store


def test_log_and_fetch_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "METRICS_DB_PATH", tmp_path / "metrics.sqlite3")
    monkeypatch.setattr(metrics_store, "METRICS_DB_PATH", tmp_path / "metrics.sqlite3")

    metrics_store.log_request(endpoint="/chat", latency_ms=123.4, model_version="v1", hallucination_score=0.1)
    metrics_store.log_request(endpoint="/classify", latency_ms=56.7)

    rows = metrics_store.fetch_recent(limit=10)

    assert len(rows) == 2
    endpoints = {r["endpoint"] for r in rows}
    assert endpoints == {"/chat", "/classify"}


def test_fetch_recent_respects_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(metrics_store, "METRICS_DB_PATH", tmp_path / "metrics.sqlite3")

    for i in range(5):
        metrics_store.log_request(endpoint="/chat", latency_ms=float(i))

    rows = metrics_store.fetch_recent(limit=3)
    assert len(rows) == 3
