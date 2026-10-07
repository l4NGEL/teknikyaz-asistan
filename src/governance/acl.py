"""Belge düzeyinde yetkilendirme (permission-aware retrieval).

Neden retrieval'dan ÖNCE filtre: yetkisiz chunk'lar yalnızca cevaptan çıkarılırsa
yine de reranker'a gider, fusion normalizasyonunu (min-max) etkiler ve LLM bağlamına
sızabilir. Burada dense/BM25 adayları füzyondan önce süzülür; yetkisiz belge hiçbir
sinyale katılmaz. Politika deny-by-default: hiçbir kurala uymayan belge `restricted`.

Kimlik (principal) asla LLM'in ürettiği bir argümandan gelmez; HTTP başlığı ya da MCP
sunucusunun ortam değişkeni gibi çağıran tarafın kontrolündeki bir kaynaktan çözülür.
"""
from __future__ import annotations

import contextvars
import json
import re
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from src.config import DATA_DIR

POLICY_PATH = DATA_DIR / "governance" / "acl_policy.json"
PRINCIPALS_PATH = DATA_DIR / "governance" / "principals.json"


@dataclass(frozen=True)
class Principal:
    user_id: str
    clearance: str = "public"
    groups: frozenset[str] = field(default_factory=frozenset)


@dataclass(frozen=True)
class DocAccess:
    classification: str
    groups: frozenset[str]


class AccessPolicy:
    def __init__(self, levels: list[str], default: dict, rules: list[dict]):
        self.levels = {name: rank for rank, name in enumerate(levels)}
        self.default = DocAccess(default["classification"], frozenset(default.get("groups", [])))
        self.rules = [
            (re.compile(r["title_regex"]), DocAccess(r["classification"], frozenset(r.get("groups", []))))
            for r in rules
        ]
        for _, access in self.rules:
            if access.classification not in self.levels:
                raise ValueError(f"bilinmeyen sınıf: {access.classification}")

    @classmethod
    def from_file(cls, path: Path = POLICY_PATH) -> AccessPolicy:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(raw["levels"], raw["default"], raw["rules"])

    def classify(self, doc_title: str) -> DocAccess:
        for pattern, access in self.rules:
            if pattern.search(doc_title or ""):
                return access
        return self.default

    def can_read(self, principal: Principal, doc_title: str) -> bool:
        access = self.classify(doc_title)
        if self.levels.get(principal.clearance, -1) < self.levels[access.classification]:
            return False
        # Grup listesi boşsa sınıf yeterli; doluysa en az bir ortak grup şart.
        return not access.groups or bool(access.groups & principal.groups)

    def filter_hits(self, principal: Principal, hits: list[dict]) -> tuple[list[dict], int]:
        allowed = [h for h in hits if self.can_read(principal, h.get("doc_title") or "")]
        return allowed, len(hits) - len(allowed)


@lru_cache(maxsize=1)
def default_policy() -> AccessPolicy:
    return AccessPolicy.from_file()


def load_principals(path: Path = PRINCIPALS_PATH) -> dict[str, Principal]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return {
        user_id: Principal(user_id, spec.get("clearance", "public"), frozenset(spec.get("groups", [])))
        for user_id, spec in raw.items()
        if not user_id.startswith("_")
    }


def resolve_principal(user_id: str | None, principals: dict[str, Principal] | None = None) -> Principal:
    """Bilinmeyen ya da boş kimlik en düşük yetkiye düşer (fail-closed)."""
    principals = principals if principals is not None else load_principals()
    if user_id and user_id in principals:
        return principals[user_id]
    return Principal(user_id or "anonymous", "public", frozenset())


# İstek kapsamındaki kimlik: /chat gibi uç noktalarda retrieve() çağrısı ajan grafiğinin
# derinlerinde yapılır; kimliği her ara katmana parametre olarak taşımak yerine istek
# başında bağlanır, retriever açık bir principal verilmemişse buradan okur.
_active_principal: contextvars.ContextVar[Principal | None] = contextvars.ContextVar("active_principal", default=None)


@contextmanager
def principal_scope(principal: Principal):
    token = _active_principal.set(principal)
    try:
        yield principal
    finally:
        _active_principal.reset(token)


def active_principal() -> Principal | None:
    return _active_principal.get()
