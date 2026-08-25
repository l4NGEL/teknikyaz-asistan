"""Chroma tabanlı kalıcı vektör veritabanı sarmalayıcısı."""
from __future__ import annotations

import chromadb

from src.config import RAG_CONFIG, VECTOR_DB_DIR
from src.rag.embeddings import embed_passages, embed_query

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=str(VECTOR_DB_DIR))
    return _client


def get_collection(name: str | None = None):
    client = _get_client()
    return client.get_or_create_collection(name or RAG_CONFIG.collection_name)


def index_chunks(chunks: list[dict], collection_name: str | None = None, batch_size: int = 64) -> int:
    """chunks: [{chunk_id, text, doc_title, doc_url, metadata}, ...] (bkz. data_prep.chunking.Chunk)"""
    collection = get_collection(collection_name)
    total = 0
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i: i + batch_size]
        texts = [c["text"] for c in batch]
        # Baslik embed edilen metne dahil edilmezse konu kimligi kayboluyor: birçok
        # şablonun govde metni jenerik yazilmis (orn. "README Sablonu"nun govdesinde
        # "readme" kelimesi hic gecmiyor, konuyu sadece baslik tasiyor).
        titled_texts = [f"{c['doc_title']}: {t}" for c, t in zip(batch, texts)]
        embeddings = embed_passages(titled_texts)
        collection.upsert(
            ids=[c["chunk_id"] for c in batch],
            embeddings=embeddings,
            documents=texts,
            metadatas=[
                {"doc_title": c["doc_title"], "doc_url": c["doc_url"], **c.get("metadata", {})}
                for c in batch
            ],
        )
        total += len(batch)
    return total


def query(text: str, top_k: int | None = None, collection_name: str | None = None) -> list[dict]:
    collection = get_collection(collection_name)
    top_k = top_k or RAG_CONFIG.top_k
    query_embedding = embed_query(text)
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)

    hits = []
    for doc, meta, dist, doc_id in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0], results["ids"][0]
    ):
        hits.append(
            {
                "chunk_id": doc_id,
                "text": doc,
                "doc_title": meta.get("doc_title"),
                "doc_url": meta.get("doc_url"),
                "similarity": 1 - dist,  # Chroma cosine distance -> benzerlik skoruna çevir
            }
        )
    return hits
