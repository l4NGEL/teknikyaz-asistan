"""TeknikYaz bilgi tabanını MCP araçları olarak sunar (Claude Desktop, Cursor, IDE ajanları).

Güvenlik kararları:
- Kimlik araç argümanı DEĞİLDİR. Model `user="admin"` diye bir argüman uyduramaz; kimlik
  sunucuyu başlatan tarafın ortam değişkeninden (TEKNIKYAZ_USER) bir kez çözülür.
- `get_document` yetkisiz ve var olmayan belge için AYNI hatayı döner; aksi halde hata
  mesajı gizli belge başlıklarının varlığını sızdıran bir kâhine (oracle) dönüşür.
- Tüm araçlar salt okunur (readOnlyHint); her çağrı denetim kaydına yazılır, dönen
  metin PII/secret maskelemeden geçer.

Çalıştırma (stdio):
    TEKNIKYAZ_USER=gelistirici python -m src.serving.mcp_server
"""
from __future__ import annotations

import json
import os
from collections.abc import Callable
from functools import lru_cache

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations
from typing_extensions import TypedDict  # pydantic, Python < 3.12 için bunu ister

from src.config import PROCESSED_DIR
from src.governance.acl import AccessPolicy, Principal, default_policy, resolve_principal
from src.governance.audit import log_access
from src.governance.pii import mask
from src.llmops.tracing import span

READ_ONLY = ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False)
NOT_FOUND = "Belge bulunamadı ya da bu belgeye erişim yetkiniz yok."


class SearchResult(TypedDict):
    doc_title: str
    chunk_id: str
    snippet: str
    score: float


class Document(TypedDict):
    doc_title: str
    classification: str
    text: str


class Identity(TypedDict):
    user: str
    clearance: str
    groups: list[str]


@lru_cache(maxsize=1)
def _chunks_by_title() -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    with open(PROCESSED_DIR / "chunks.jsonl", encoding="utf-8") as f:
        for line in f:
            chunk = json.loads(line)
            grouped.setdefault(chunk["doc_title"], []).append(chunk)
    for chunks in grouped.values():
        chunks.sort(key=lambda c: (c.get("metadata") or {}).get("chunk_index", 0))
    return grouped


def _default_retrieve(question: str, final_k: int, principal: Principal) -> tuple[list[dict], int]:
    from src.rag.retriever import retrieve_with_report

    return retrieve_with_report(question, final_k=final_k, principal=principal, hierarchical=False)


def build_server(
    principal: Principal,
    retrieve_fn: Callable[[str, int, Principal], tuple[list[dict], int]] = _default_retrieve,
    documents: Callable[[], dict[str, list[dict]]] = _chunks_by_title,
    policy: AccessPolicy | None = None,
    audit: Callable[..., object] = log_access,
) -> MCPServer:
    policy = policy or default_policy()
    server = MCPServer(
        name="teknikyaz",
        instructions=(
            "Türkçe teknik dokümantasyon (README, API, kurulum, mimari, SSS) bilgi tabanı. "
            "Önce search_docs ile ara, gerekirse get_document ile belgenin tamamını oku. "
            "Sonuçlar yalnızca bu sunucuyu başlatan kullanıcının okuma yetkisi olan belgelerdir."
        ),
    )

    @server.tool(annotations=READ_ONLY)
    def search_docs(query: str, top_k: int = 5) -> list[SearchResult]:
        """Hibrit (dense + BM25 + RRF) arama. Yalnızca okuma yetkiniz olan belgeleri döndürür."""
        top_k = max(1, min(int(top_k), 20))
        with span("mcp.search_docs", user=principal.user_id, query=query):
            hits, denied = retrieve_fn(query, top_k, principal)
        audit(principal.user_id, "mcp.search_docs", query, [h["chunk_id"] for h in hits], denied)
        return [
            {
                "doc_title": h.get("doc_title") or "",
                "chunk_id": h["chunk_id"],
                "snippet": mask((h.get("text") or "")[:300]),
                "score": round(float(h.get("fusion_score", 0.0)), 4),
            }
            for h in hits
        ]

    @server.tool(annotations=READ_ONLY)
    def get_document(doc_title: str) -> Document:
        """Bir belgenin tam metnini başlığıyla getirir (search_docs sonucundaki doc_title)."""
        chunks = documents().get(doc_title)
        allowed = chunks is not None and policy.can_read(principal, doc_title)
        audit(principal.user_id, "mcp.get_document", doc_title, [doc_title] if allowed else [], 0 if allowed else 1)
        if not allowed:
            raise ToolError(NOT_FOUND)
        return {
            "doc_title": doc_title,
            "classification": policy.classify(doc_title).classification,
            "text": mask("\n\n".join(c["text"] for c in chunks)),
        }

    @server.tool(annotations=READ_ONLY)
    def whoami() -> Identity:
        """Bu oturumun kimliği ve erişim düzeyi (hangi belge sınıflarını görebileceğiniz)."""
        return {
            "user": principal.user_id,
            "clearance": principal.clearance,
            "groups": sorted(principal.groups),
        }

    return server


def main() -> None:
    principal = resolve_principal(os.environ.get("TEKNIKYAZ_USER"))
    build_server(principal).run("stdio")


if __name__ == "__main__":
    main()
