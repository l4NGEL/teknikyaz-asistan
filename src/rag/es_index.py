"""Elasticsearch arka ucu: Türkçe analizörlü BM25 + dense_vector kNN, tek indekste.

Neden: `rank_bm25` her sorguda tüm korpusu bellekte puanlıyor ve Türkçe ekleri
tanımıyor ("şablonunda" ile "şablon" farklı token). Elasticsearch'ün `turkish`
analizörü (Türkçe küçük harf + stopword + Snowball kök bulucu) bu ekleri soyuyor;
aynı indekste HNSW tabanlı vektör araması da yapılabildiği için Chroma'ya ayrı bir
depo gerekmiyor.

Üç metin alanı bilerek tutuluyor, böylece analizörün etkisi aynı motor ve aynı
BM25 parametreleriyle ayrıca ölçülebiliyor (bkz. scripts/run_es_eval.py):
- `title_text_std`: standard analizör (kök bulma yok).
- `title_text`: yerleşik `turkish` analizörü (Snowball). Ölçümde tutarsız kök
  bulduğu görüldü: "Şablonu" -> "şablo" ama "şablonunda" -> "şablon", yani şablon
  belgesi kendi adını içeren sorguda bile eşleşmiyor.
- `title_text_f5`: Türkçe küçük harf + kesme işareti temizliği + her token'ı ilk 5
  harfe kesme (Türkçe bilgi erişiminde bilinen "F5" yaklaşımı). Eklemeli yapıda
  kökün çoğu ilk 5 harfte kaldığı için "şablonu" / "şablonunda" ikisi de "şablo"
  olur. Varsayılan BM25 alanı budur.

Dönüş şekli `bm25_index.query` / `vector_store.query` ile aynıdır, `retriever.py`
ikisini değiştirmeden birleştirebilir.
"""
from __future__ import annotations

import json
import os

from elasticsearch import Elasticsearch, helpers

from src.config import PROCESSED_DIR, RAG_CONFIG
from src.rag.embeddings import embed_passages, embed_query

ES_URL = os.environ.get("ES_URL", "http://localhost:9200")
INDEX_NAME = os.environ.get("ES_INDEX", f"{RAG_CONFIG.collection_name}_es")
EMBED_DIMS = 1024  # intfloat/multilingual-e5-large
DEFAULT_BM25_FIELD = "title_text_f5"

_client: Elasticsearch | None = None


def get_client() -> Elasticsearch:
    global _client
    if _client is None:
        _client = Elasticsearch(ES_URL, request_timeout=30)
    return _client


def index_mapping(dims: int = EMBED_DIMS) -> dict:
    return {
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0,
            "analysis": {
                "filter": {
                    "tr_lowercase": {"type": "lowercase", "language": "turkish"},
                    "tr_stop": {"type": "stop", "stopwords": "_turkish_"},
                    "prefix5": {"type": "truncate", "length": 5},
                },
                "analyzer": {
                    "turkish_f5": {
                        "type": "custom",
                        "tokenizer": "standard",
                        "filter": ["apostrophe", "tr_lowercase", "tr_stop", "prefix5"],
                    }
                },
            },
        },
        "mappings": {
            "properties": {
                "chunk_id": {"type": "keyword"},
                "doc_title": {"type": "text", "analyzer": "turkish", "fields": {"raw": {"type": "keyword"}}},
                "doc_url": {"type": "keyword", "index": False},
                "text": {"type": "text", "analyzer": "turkish"},
                # Başlık gövdeye eklenmezse konu kimliği kayboluyor (bkz. bm25_index.py).
                "title_text": {"type": "text", "analyzer": "turkish"},
                "title_text_std": {"type": "text", "analyzer": "standard"},
                "title_text_f5": {"type": "text", "analyzer": "turkish_f5"},
                "embedding": {"type": "dense_vector", "dims": dims, "index": True, "similarity": "cosine"},
            }
        },
    }


def load_chunks() -> list[dict]:
    with open(PROCESSED_DIR / "chunks.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_actions(chunks: list[dict], embeddings: list[list[float]], index: str = INDEX_NAME):
    for chunk, vector in zip(chunks, embeddings):
        title_text = f"{chunk['doc_title']} {chunk['text']}"
        yield {
            "_index": index,
            "_id": chunk["chunk_id"],
            "_source": {
                "chunk_id": chunk["chunk_id"],
                "doc_title": chunk["doc_title"],
                "doc_url": chunk.get("doc_url"),
                "text": chunk["text"],
                "title_text": title_text,
                "title_text_std": title_text,
                "title_text_f5": title_text,
                "embedding": vector,
            },
        }


def index_chunks(chunks: list[dict] | None = None, index: str = INDEX_NAME, recreate: bool = True) -> int:
    """Chunk'ları gömüp indeksler; Chroma ile aynı "başlık: metin" pasaj biçimi."""
    es = get_client()
    chunks = chunks if chunks is not None else load_chunks()
    if recreate and es.indices.exists(index=index):
        es.indices.delete(index=index)
    if not es.indices.exists(index=index):
        es.indices.create(index=index, **index_mapping())
    embeddings = embed_passages([f"{c['doc_title']}: {c['text']}" for c in chunks])
    ok, _ = helpers.bulk(es, build_actions(chunks, embeddings, index), refresh="wait_for")
    return ok


def _to_hit(doc: dict, score_key: str) -> dict:
    src = doc["_source"]
    return {
        "chunk_id": src["chunk_id"],
        "text": src["text"],
        "doc_title": src["doc_title"],
        "doc_url": src.get("doc_url"),
        score_key: float(doc["_score"]),
    }


def bm25_query(text: str, top_k: int = 20, field: str = DEFAULT_BM25_FIELD) -> dict:
    return {"query": {"match": {field: {"query": text}}}, "size": top_k, "_source": {"excludes": ["embedding"]}}


def knn_query(vector: list[float], top_k: int = 20) -> dict:
    return {
        "knn": {"field": "embedding", "query_vector": vector, "k": top_k, "num_candidates": max(100, top_k * 5)},
        "size": top_k,
        "_source": {"excludes": ["embedding"]},
    }


def query_bm25(text: str, top_k: int = 20, field: str = DEFAULT_BM25_FIELD, index: str = INDEX_NAME) -> list[dict]:
    resp = get_client().search(index=index, **bm25_query(text, top_k, field))
    return [_to_hit(d, "bm25_score") for d in resp["hits"]["hits"]]


def query_dense(text: str, top_k: int = 20, index: str = INDEX_NAME) -> list[dict]:
    resp = get_client().search(index=index, **knn_query(embed_query(text), top_k))
    hits = []
    for doc in resp["hits"]["hits"]:
        hit = _to_hit(doc, "es_score")
        # ES cosine skoru (1 + cos) / 2 döndürür; Chroma ile karşılaştırılabilir olsun diye cos'a çevir.
        hit["similarity"] = 2 * hit.pop("es_score") - 1
        hits.append(hit)
    return hits
