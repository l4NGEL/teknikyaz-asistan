"""BM25 (lexical/anahtar-kelime) tabanlı retrieval.

Neden gerekli: saf embedding tabanlı (dense) retrieval, "README" kelimesi hem soruda
hem de en alakalı dokümanın başlığında birebir geçse bile bu dokümanı kaçırabiliyor —
embedding modelleri anlamsal benzerliği ölçer, birebir kelime eşleşmesini değil. BM25
tam tersi bir sinyal sağlar: nadir/ayırt edici kelimelerin birebir eşleşmesine yüksek
ağırlık verir. İkisini `retriever.py`'de birleştirmek (hybrid retrieval), her iki
sinyalin de güçlü olduğu durumları kapsar.
"""
from __future__ import annotations

import json
import re

from rank_bm25 import BM25Okapi

from src.config import PROCESSED_DIR

_bm25: BM25Okapi | None = None
_chunks: list[dict] | None = None

# Türkçe'de en sık geçen, ayırt edici olmayan işlev kelimeleri. Soru cümlelerinde
# ("Bir README'de HANGİ bölümler OLMALI?") bunlar filtrelenmezse BM25'in nadir/ayırt
# edici kelimelere (örn. "readme") verdiği ağırlık sulanıyor.
_STOPWORDS = frozenset({
    "acaba", "ama", "bir", "bu", "da", "de", "en", "gibi", "hangi", "hem", "her",
    "ile", "içinde", "ise", "için", "ki", "kim", "mi", "mı", "mu", "mü", "ne",
    "nedir", "nasıl", "niye", "o", "olarak", "olmalı", "ve", "veya", "ya", "yani",
})


def _tokenize(text: str) -> list[str]:
    tokens = re.findall(r"\w+", text.lower())
    return [t for t in tokens if t not in _STOPWORDS]


def _load_index() -> tuple[BM25Okapi, list[dict]]:
    global _bm25, _chunks
    if _bm25 is None:
        chunks = []
        with open(PROCESSED_DIR / "chunks.jsonl", encoding="utf-8") as f:
            for line in f:
                chunks.append(json.loads(line))
        # Baslik metne dahil edilmezse anahtar-kelime sinyali kayboluyor: birçok şablonun
        # govde metni jenerik yazilmis (orn. "README Sablonu"nun govdesinde "readme" kelimesi
        # hic gecmiyor, konu kimligini sadece baslik tasiyor) -- baslik + govde birlikte
        # indeksleniyor ki "README" gibi sorgular dogru chunk'i bulabilsin.
        _bm25 = BM25Okapi([_tokenize(f"{c['doc_title']} {c['text']}") for c in chunks])
        _chunks = chunks
    return _bm25, _chunks


def query(text: str, top_k: int = 20) -> list[dict]:
    """Dense `vector_store.query`'yle aynı sözlük şeklini döndürür (chunk_id, text,
    doc_title, doc_url, + bm25_score) ki `retriever.py` ikisini sorunsuz birleştirebilsin."""
    bm25, chunks = _load_index()
    scores = bm25.get_scores(_tokenize(text))
    ranked = sorted(zip(chunks, scores), key=lambda pair: pair[1], reverse=True)[:top_k]
    return [
        {
            "chunk_id": c["chunk_id"],
            "text": c["text"],
            "doc_title": c["doc_title"],
            "doc_url": c.get("doc_url"),
            "bm25_score": float(s),
        }
        for c, s in ranked
        if s > 0
    ]
