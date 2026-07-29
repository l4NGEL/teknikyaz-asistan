"""Çok dilli embedding sarmalayıcısı.

E5 ailesi modeller, sorgu (query) ve pasaj (passage) metinlerinin farklı önceklerle
("query: " / "passage: ") kodlanmasını bekler — bu ayrımı atlamak retrieval kalitesini
belirgin şekilde düşürür, bu yüzden burada API seviyesinde zorunlu kılıyoruz.
"""
from __future__ import annotations

from sentence_transformers import SentenceTransformer

from src.config import MODEL_CONFIG

_model = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_CONFIG.embedding_model)
    return _model


def embed_passages(texts: list[str]) -> list[list[float]]:
    model = _get_model()
    prefixed = [f"passage: {t}" for t in texts]
    return model.encode(prefixed, normalize_embeddings=True, show_progress_bar=False).tolist()


def embed_query(text: str) -> list[float]:
    model = _get_model()
    return model.encode(f"query: {text}", normalize_embeddings=True, show_progress_bar=False).tolist()
