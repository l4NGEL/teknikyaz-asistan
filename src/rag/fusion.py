"""Hibrit skor birleştirme: RRF (dense + BM25) + isteğe bağlı rerank + başlık sinyali.

Reranker tek başına sıralamayı devralırsa lexical isabet (başlıkta "README") gömülüyor;
bu, Diagnis ölçümündeki miss'lerin ana kaynağıydı. RRF her iki aday listesinin
sıralamasını korur, rerank skoru ise *ek* bir sinyal olur, tek hakem değil.
"""
from __future__ import annotations

import re

RRF_K = 60
TEMPLATE_HINTS = (
    "şablon",
    "sablon",
    "nasıl yazılır",
    "nasil yazilir",
    "hangi bölüm",
    "hangi bolum",
    "nasıl yapılandır",
    "nasil yapilandir",
    "neleri içermeli",
    "neleri kapsar",
    "hangi adımlar",
    "hangi adimlar",
    "hangi soru",
    "belgesi hangi",
    "kapsamalı",
    "kapsamali",
)

_STOPWORDS = frozenset({
    "acaba", "ama", "bir", "bu", "da", "de", "en", "gibi", "hangi", "hem", "her",
    "ile", "içinde", "ise", "için", "ki", "kim", "mi", "mı", "mu", "mü", "ne",
    "nedir", "nasıl", "niye", "o", "olarak", "olmalı", "ve", "veya", "ya", "yani",
})


def tokenize(text: str) -> list[str]:
    tokens = re.findall(r"\w+", (text or "").lower())
    return [t for t in tokens if t not in _STOPWORDS]


def minmax_norm(values: list[float]) -> list[float]:
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi - lo < 1e-12:
        return [0.0 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


def rrf_scores(ranked_ids: list[str], k: int = RRF_K) -> dict[str, float]:
    return {doc_id: 1.0 / (k + rank + 1) for rank, doc_id in enumerate(ranked_ids)}


def title_overlap(question: str, title: str) -> float:
    q = set(tokenize(question))
    t = set(tokenize(title))
    if not q or not t:
        return 0.0
    return len(q & t) / len(q)


def wants_template(question: str) -> bool:
    lowered = (question or "").lower()
    return any(hint in lowered for hint in TEMPLATE_HINTS)


def template_boost(question: str, title: str) -> float:
    """Şablon isteniyorsa başlığında 'şablon' geçen ve soruyla kelime paylaşan belgeye tam boost."""
    if not wants_template(question):
        return 0.0
    lowered = (title or "").lower()
    if "şablon" not in lowered and "sablon" not in lowered:
        return 0.0
    if title_overlap(question, title) > 0:
        return 1.0
    return 0.15


def merge_hit_dicts(dense_hits: list[dict], bm25_hits: list[dict]) -> dict[str, dict]:
    """Aynı chunk_id için dense + BM25 alanlarını birleştirir (setdefault BM25 skorunu düşürüyordu)."""
    merged: dict[str, dict] = {}
    for hit in dense_hits:
        merged[hit["chunk_id"]] = dict(hit)
    for hit in bm25_hits:
        cid = hit["chunk_id"]
        if cid in merged:
            merged[cid]["bm25_score"] = hit.get("bm25_score", 0.0)
            if not merged[cid].get("text"):
                merged[cid]["text"] = hit.get("text")
            if not merged[cid].get("doc_title"):
                merged[cid]["doc_title"] = hit.get("doc_title")
        else:
            merged[cid] = dict(hit)
    return merged


def fuse_candidates(
    question: str,
    dense_hits: list[dict],
    bm25_hits: list[dict],
    rerank_scores: dict[str, float] | None = None,
) -> list[dict]:
    merged = merge_hit_dicts(dense_hits, bm25_hits)
    if not merged:
        return []

    rrf_dense = rrf_scores([h["chunk_id"] for h in dense_hits])
    rrf_bm25 = rrf_scores([h["chunk_id"] for h in bm25_hits])
    ids = list(merged.keys())
    rrf = [rrf_dense.get(i, 0.0) + rrf_bm25.get(i, 0.0) for i in ids]
    titles = [title_overlap(question, merged[i].get("doc_title") or "") for i in ids]
    templates = [template_boost(question, merged[i].get("doc_title") or "") for i in ids]
    rrf_n = minmax_norm(rrf)
    title_n = minmax_norm(titles)

    if rerank_scores:
        rerank_raw = [float(rerank_scores.get(i, 0.0)) for i in ids]
        rerank_n = minmax_norm(rerank_raw)
        # Rerank ek sinyal; lexical RRF + başlık, genel reranker'ın şablon isabetini
        # gömmesini engellemek için ağırlığı kasıtlı olarak düşük tutulur.
        weights = (0.20, 0.40, 0.20, 0.20)  # rerank, rrf, title, template
    else:
        rerank_n = [0.0] * len(ids)
        weights = (0.0, 0.60, 0.25, 0.15)

    ranked = []
    for i, doc_id in enumerate(ids):
        item = dict(merged[doc_id])
        score = (
            weights[0] * rerank_n[i]
            + weights[1] * rrf_n[i]
            + weights[2] * title_n[i]
            + weights[3] * templates[i]
        )
        if rerank_scores and doc_id in rerank_scores:
            item["rerank_score"] = float(rerank_scores[doc_id])
        item["fusion_score"] = float(score)
        item["rrf_score"] = float(rrf[i])
        item["title_overlap"] = float(titles[i])
        ranked.append(item)

    ranked.sort(key=lambda c: c["fusion_score"], reverse=True)
    return ranked
