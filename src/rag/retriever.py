"""Retrieval katmanı: dense + BM25 adayları RRF ile birleştirilir, rerank ek sinyaldir.

Neden reranking tek başına yetmez: embedding benzerliği kabadır; cross-encoder daha
isabetli olabilir ama genel amaçlı çok dilli bir reranker dar teknik Türkçe başlık
eşleşmesini (README Şablonu) gömebiliyor. Dense ve BM25 sıralarını Reciprocal Rank
Fusion ile koruyup rerank + başlık örtüşmesini üzerine ekliyoruz.

`principal` verilirse yetkisiz adaylar füzyondan ve reranker'dan ÖNCE düşer
(bkz. src/governance/acl.py); aday havuzu süzme kaybını telafi için genişletilir.
"""
from __future__ import annotations

from sentence_transformers import CrossEncoder

from src.config import RAG_CONFIG
from src.governance.acl import AccessPolicy, Principal, active_principal, default_policy
from src.llmops.tracing import span
from src.rag import bm25_index, vector_store
from src.rag.fusion import fuse_candidates
from src.rag.hierarchy import expand_hits

_reranker = None
RERANKER_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"


def _get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(RERANKER_MODEL)
    return _reranker


def retrieve(question: str, **kwargs) -> list[dict]:
    return retrieve_with_report(question, **kwargs)[0]


def retrieve_with_report(
    question: str,
    candidate_k: int = 20,
    final_k: int = 5,
    rerank: bool = True,
    hierarchical: bool | None = None,
    principal: Principal | None = None,
    policy: AccessPolicy | None = None,
) -> tuple[list[dict], int]:
    """(hits, yetki nedeniyle elenen aday sayısı)."""
    principal = principal or active_principal()
    denied = 0
    with span("retrieve", question=question, final_k=final_k) as root:
        # ACL süzmesi aday sayısını düşürür; havuz genişletilmezse kısıtlı kullanıcıda
        # izinli ama 20. sıranın altındaki belge hiç aday olamaz.
        pool_k = candidate_k * 3 if principal is not None else candidate_k
        with span("dense"):
            dense_hits = vector_store.query(question, top_k=pool_k)
        with span("bm25"):
            bm25_hits = bm25_index.query(question, top_k=pool_k)

        if principal is not None:
            with span("acl_filter", user=principal.user_id, clearance=principal.clearance) as acl_span:
                policy = policy or default_policy()
                dense_hits, dense_denied = policy.filter_hits(principal, dense_hits)
                bm25_hits, bm25_denied = policy.filter_hits(principal, bm25_hits)
                dense_hits, bm25_hits = dense_hits[:candidate_k], bm25_hits[:candidate_k]
                denied = dense_denied + bm25_denied
                acl_span.set(denied_candidates=denied)
            root.set(denied_candidates=denied)

        rerank_scores = None
        if rerank:
            by_id = {}
            for hit in dense_hits + bm25_hits:
                by_id.setdefault(hit["chunk_id"], hit)
            if by_id:
                with span("rerank", pairs=len(by_id)):
                    reranker = _get_reranker()
                    ordered = list(by_id.values())
                    pairs = [(question, c.get("text") or "") for c in ordered]
                    scores = reranker.predict(pairs)
                    rerank_scores = {c["chunk_id"]: float(s) for c, s in zip(ordered, scores)}

        with span("fusion"):
            ranked = fuse_candidates(question, dense_hits, bm25_hits, rerank_scores=rerank_scores)
        hits = ranked[:final_k]

        if hierarchical if hierarchical is not None else RAG_CONFIG.hierarchical_expand:
            hits = expand_hits(hits)
        root.set(returned=[h["chunk_id"] for h in hits])
        return hits, denied
