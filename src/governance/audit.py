"""Değiştirilemez (tamper-evident) denetim izi: kim, ne sordu, hangi belgeler döndü.

Her kayıt bir önceki kaydın SHA-256 özetini taşır (hash zinciri). Aradaki bir satır
silinir ya da düzenlenirse `verify()` kırılan ilk satırı gösterir. Sorgu metni yazılmadan
önce PII/secret maskelemeden geçer; denetim kaydı kişisel veri deposuna dönüşmez.
"""
from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path

from src.config import DATA_DIR
from src.governance.pii import redact

AUDIT_PATH = DATA_DIR / "logs" / "audit.jsonl"
GENESIS = "0" * 64
_lock = threading.Lock()


def _digest(record: dict) -> str:
    payload = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _last_hash(path: Path) -> str:
    if not path.exists():
        return GENESIS
    last = None
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                last = line
    return json.loads(last)["hash"] if last else GENESIS


def log_access(
    user_id: str,
    action: str,
    query: str,
    returned: list[str],
    denied_count: int = 0,
    path: Path | None = None,
) -> dict:
    path = path or AUDIT_PATH
    redaction = redact(query)
    with _lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "ts": round(time.time(), 3),
            "user": user_id,
            "action": action,
            "query": redaction.text,
            "pii_masked": dict(redaction.counts),
            "returned": returned,
            "denied_count": denied_count,
            "prev": _last_hash(path),
        }
        record["hash"] = _digest(record)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record


def verify(path: Path | None = None) -> tuple[bool, int | None]:
    """(geçerli_mi, kırılan_ilk_satır_no). Dosya yoksa zincir boş ve geçerlidir."""
    path = path or AUDIT_PATH
    if not path.exists():
        return True, None
    prev = GENESIS
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            stored = record.pop("hash", None)
            if record.get("prev") != prev or _digest(record) != stored:
                return False, lineno
            prev = stored
    return True, None
