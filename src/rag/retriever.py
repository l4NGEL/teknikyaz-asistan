"""Retrieval katmanı: vector store'dan geniş bir aday seti çeker, ardından cross-encoder
ile yeniden sıralar (reranking).

Neden reranking: embedding benzerliği (bi-encoder) hızlıdır ama kabadır — soru ve pasajı
ayrı ayrı vektöre sıkıştırır. Cross-encoder, soru+pasaj çiftini birlikte okuyup çok daha
isabetli bir alaka skoru üretir. İkisini birleştirmek (retrieve-then-rerank) endüstri
standardı bir RAG deseni.
"""
from __future__ import annotations

from sentence_transformers import CrossEncoder

from src.rag import vector_store

_reranker = None
RERANKER_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"  # çok dilli, Türkçe dahil


def _get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(RERANKER_MODEL)
    return _reranker


def retrieve(question: str, candidate_k: int = 20, final_k: int = 5, rerank: bool = True) -> list[dict]:
    candidates = vector_store.query(question, top_k=candidate_k)
    if not candidates:
        return []

    if not rerank:
        return candidates[:final_k]

    reranker = _get_reranker()
    pairs = [(question, c["text"]) for c in candidates]
    scores = reranker.predict(pairs)
    for c, s in zip(candidates, scores):
        c["rerank_score"] = float(s)

    candidates.sort(key=lambda c: c["rerank_score"], reverse=True)
    return candidates[:final_k]
