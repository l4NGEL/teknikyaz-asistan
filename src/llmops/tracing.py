"""Hafif, bağımlılıksız span tabanlı tracing: istek → retrieve → acl → rerank → generate.

`metrics_store` istek başına tek satır tutar (toplam gecikme). Gecikmenin NEREDE
harcandığını görmek için her aşama ayrı bir span'dır; span'ler `contextvars` ile
iç içe bağlanır, istek bitince tek bir trace kaydı olarak JSONL'e yazılır. Alan adları
OpenTelemetry'e yakın tutuldu (trace_id, span_id, parent_id, attributes, status) ki
ileride bir OTLP exporter'a taşımak yalnızca `_export`'u değiştirmek olsun.
Öznitelikler yazılmadan önce PII/secret maskelemeden geçer.
"""
from __future__ import annotations

import contextvars
import json
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path

from src.config import DATA_DIR
from src.governance.pii import mask

TRACES_PATH = DATA_DIR / "logs" / "traces.jsonl"

_current_span: contextvars.ContextVar[Span | None] = contextvars.ContextVar("current_span", default=None)
_lock = threading.Lock()


@dataclass
class Span:
    name: str
    trace_id: str
    span_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    parent_id: str | None = None
    start: float = field(default_factory=time.perf_counter)
    duration_ms: float | None = None
    status: str = "ok"
    attributes: dict = field(default_factory=dict)
    children: list[Span] = field(default_factory=list)

    def set(self, **attrs) -> None:
        for key, value in attrs.items():
            self.attributes[key] = mask(value) if isinstance(value, str) else value

    def to_dict(self) -> dict:
        data = asdict(self)
        data.pop("start")
        return data


@contextmanager
def span(name: str, path: Path | None = None, **attrs):
    """Kök span (ebeveyni yoksa) kapanınca tüm ağaç tek trace olarak dışa yazılır."""
    parent = _current_span.get()
    current = Span(
        name=name,
        trace_id=parent.trace_id if parent else uuid.uuid4().hex,
        parent_id=parent.span_id if parent else None,
    )
    current.set(**attrs)
    if parent:
        parent.children.append(current)
    token = _current_span.set(current)
    try:
        yield current
    except Exception as exc:
        current.status = "error"
        current.set(error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        current.duration_ms = round((time.perf_counter() - current.start) * 1000, 3)
        _current_span.reset(token)
        if parent is None:
            _export(current, path or TRACES_PATH)


def current_span() -> Span | None:
    return _current_span.get()


def annotate(**attrs) -> None:
    """Aktif bir span varsa öznitelik ekler; yoksa sessizce geçer (tracing opsiyonel)."""
    active = _current_span.get()
    if active is not None:
        active.set(**attrs)


def _export(root: Span, path: Path) -> None:
    with _lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(root.to_dict(), ensure_ascii=False) + "\n")


def load_traces(path: Path | None = None, limit: int | None = None) -> list[dict]:
    path = path or TRACES_PATH
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return rows[-limit:] if limit else rows


def stage_latency_summary(traces: list[dict]) -> dict[str, dict]:
    """Aşama adı başına p50/p95/max gecikme: hangi adım pahalı sorusunun cevabı."""
    by_name: dict[str, list[float]] = {}

    def walk(node: dict) -> None:
        if node.get("duration_ms") is not None:
            by_name.setdefault(node["name"], []).append(node["duration_ms"])
        for child in node.get("children", []):
            walk(child)

    for trace in traces:
        walk(trace)

    def pct(values: list[float], q: float) -> float:
        ordered = sorted(values)
        idx = min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))
        return ordered[idx]

    return {
        name: {"count": len(v), "p50_ms": pct(v, 0.5), "p95_ms": pct(v, 0.95), "max_ms": max(v)}
        for name, v in by_name.items()
    }
