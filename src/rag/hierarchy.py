"""Hiyerarşik RAG: child chunk retrieve edilir, üretim parent belgede genişletilir.

Child (küçük, ~256 token) parçalar retrieval isabetini artırır; aynı belgenin kardeş
parçalarını birleştirmek (parent) ise kesilmiş cümle/bölüm bağlamını generation'a
geri verir. Bu, Diagnis ilanındaki "hiyerarşik RAG / chunk stratejisi" maddesinin
bu projedeki somut karşılığıdır.
"""
from __future__ import annotations

import json
from functools import lru_cache

from src.config import PROCESSED_DIR


def parent_id_of(chunk: dict) -> str:
    meta = chunk.get("metadata") or {}
    if isinstance(meta, dict) and meta.get("parent_id"):
        return str(meta["parent_id"])
    chunk_id = str(chunk.get("chunk_id") or "")
    if "::" in chunk_id:
        return chunk_id.rsplit("::", 1)[0]
    return str(chunk.get("doc_title") or chunk_id)


def _chunk_index(chunk: dict) -> int:
    meta = chunk.get("metadata") or {}
    if isinstance(meta, dict) and "chunk_index" in meta:
        try:
            return int(meta["chunk_index"])
        except (TypeError, ValueError):
            pass
    chunk_id = str(chunk.get("chunk_id") or "")
    if "::" in chunk_id:
        tail = chunk_id.rsplit("::", 1)[-1]
        if tail.isdigit():
            return int(tail)
    return 0


@lru_cache(maxsize=1)
def _load_corpus_chunks() -> tuple[dict, ...]:
    path = PROCESSED_DIR / "chunks.jsonl"
    if not path.exists():
        return ()
    chunks = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return tuple(chunks)


def siblings_by_parent(parent_id: str, corpus: list[dict] | None = None) -> list[dict]:
    source = list(corpus) if corpus is not None else list(_load_corpus_chunks())
    matches = [c for c in source if parent_id_of(c) == parent_id]
    matches.sort(key=_chunk_index)
    return matches


def expand_hits(hits: list[dict], corpus: list[dict] | None = None) -> list[dict]:
    """Her child hit'e parent metnini ve kardeş sayısını ekler; sıralamayı korur."""
    if not hits:
        return []

    source = list(corpus) if corpus is not None else list(_load_corpus_chunks())
    by_parent: dict[str, list[dict]] = {}
    for chunk in source:
        by_parent.setdefault(parent_id_of(chunk), []).append(chunk)
    for pid in by_parent:
        by_parent[pid].sort(key=_chunk_index)

    expanded = []
    for hit in hits:
        pid = parent_id_of(hit)
        siblings = by_parent.get(pid) or [hit]
        parent_text = "\n".join(s.get("text") or "" for s in siblings).strip() or hit.get("text") or ""
        item = dict(hit)
        item["parent_id"] = pid
        item["parent_text"] = parent_text
        item["sibling_count"] = len(siblings)
        item["level"] = "child"
        expanded.append(item)
    return expanded
