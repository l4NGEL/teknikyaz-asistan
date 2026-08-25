"""Retrieval katmanı: dense + BM25 adayları RRF ile birleştirilir, rerank ek sinyaldir.

Neden reranking tek başına yetmez: embedding benzerliği kabadır; cross-encoder daha
isabetli olabilir ama genel amaçlı çok dilli bir reranker dar teknik Türkçe başlık
eşleşmesini (README Şablonu) gömebiliyor. Dense ve BM25 sıralarını Reciprocal Rank
Fusion ile koruyup rerank + başlık örtüşmesini üzerine ekliyoruz.
"""
from __future__ import annotations

from sentence_transformers import CrossEncoder

from src.config import RAG_CONFIG
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


def retrieve(
    question: str,
    candidate_k: int = 20,
    final_k: int = 5,
    rerank: bool = True,
    hierarchical: bool | None = None,
) -> list[dict]:
    dense_hits = vector_store.query(question, top_k=candidate_k)
    bm25_hits = bm25_index.query(question, top_k=candidate_k)

    rerank_scores = None
    if rerank:
        by_id = {}
        for hit in dense_hits + bm25_hits:
            by_id.setdefault(hit["chunk_id"], hit)
        if by_id:
            reranker = _get_reranker()
            ordered = list(by_id.values())
            pairs = [(question, c.get("text") or "") for c in ordered]
            scores = reranker.predict(pairs)
            rerank_scores = {c["chunk_id"]: float(s) for c, s in zip(ordered, scores)}

    ranked = fuse_candidates(question, dense_hits, bm25_hits, rerank_scores=rerank_scores)
    hits = ranked[:final_k]

    if hierarchical if hierarchical is not None else RAG_CONFIG.hierarchical_expand:
        hits = expand_hits(hits)
    return hits
