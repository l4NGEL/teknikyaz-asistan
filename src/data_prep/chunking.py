"""RAG için doküman parçalama (chunking).

Neden token-bazlı ve overlap'li: LLM'lerin context penceresi token cinsinden ölçülür,
karakter bazlı bölme yanıltıcıdır. Overlap (örtüşme) olmazsa bir cümlenin ortasında
bölünen bilgi retrieval'da kaybolabilir.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from transformers import AutoTokenizer

from src.config import MODEL_CONFIG, PROCESSED_DIR, RAG_CONFIG, RAW_DIR

_tokenizer = None


def _get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        # Embedding modelinin tokenizer'ını kullanmak, chunk boyutunu retrieval modeliyle
        # tutarlı tutar.
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_CONFIG.embedding_model)
    return _tokenizer


@dataclass
class Chunk:
    doc_title: str
    doc_url: str
    chunk_id: str
    text: str
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "doc_title": self.doc_title,
            "doc_url": self.doc_url,
            "chunk_id": self.chunk_id,
            "text": self.text,
            "metadata": self.metadata,
        }


def chunk_text(text: str, chunk_size: int | None = None, overlap: int | None = None) -> list[str]:
    tokenizer = _get_tokenizer()
    chunk_size = chunk_size or RAG_CONFIG.chunk_size_tokens
    overlap = overlap if overlap is not None else RAG_CONFIG.chunk_overlap_tokens

    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if not token_ids:
        return []

    chunks = []
    start = 0
    step = max(chunk_size - overlap, 1)
    while start < len(token_ids):
        window = token_ids[start: start + chunk_size]
        chunks.append(tokenizer.decode(window, skip_special_tokens=True).strip())
        if start + chunk_size >= len(token_ids):
            break
        start += step
    return [c for c in chunks if c]


def chunk_corpus(corpus_path=None) -> list[Chunk]:
    corpus_path = corpus_path or (RAW_DIR / "corpus.jsonl")
    chunks: list[Chunk] = []
    with open(corpus_path, encoding="utf-8") as f:
        for doc in f:
            doc = json.loads(doc)
            pieces = chunk_text(doc["text"])
            for i, piece in enumerate(pieces):
                chunks.append(
                    Chunk(
                        doc_title=doc["title"],
                        doc_url=doc["url"],
                        chunk_id=f"{doc['title']}::{i}",
                        text=piece,
                        metadata={"chunk_index": i, "total_chunks": len(pieces)},
                    )
                )
    return chunks


def save_chunks(chunks: list[Chunk], filename: str = "chunks.jsonl") -> str:
    out_path = PROCESSED_DIR / filename
    with out_path.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")
    return str(out_path)


if __name__ == "__main__":
    chunks = chunk_corpus()
    path = save_chunks(chunks)
    print(f"{len(chunks)} chunk kaydedildi -> {path}")
