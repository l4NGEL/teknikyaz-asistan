import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pytest


@pytest.fixture(autouse=True)
def _isolate_runtime_logs(tmp_path, monkeypatch):
    # retrieve()/API çağrıları trace ve denetim kaydı yazar; testler gerçek data/logs'u kirletmesin.
    from src.governance import audit
    from src.llmops import tracing

    monkeypatch.setattr(tracing, "TRACES_PATH", tmp_path / "traces.jsonl")
    monkeypatch.setattr(audit, "AUDIT_PATH", tmp_path / "audit.jsonl")
