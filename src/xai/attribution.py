"""Açıklanabilirlik: bir cevabın hangi kaynak chunk'lardan/cümlelerden geldiğini gösterir.

İki seviyeli attribution sunuyoruz:
1. Chunk-seviyesi: her retrieved context'in cevaba embedding benzerliği (hangi kaynak
   en çok katkı sağladı).
2. Cümle-seviyesi: cevaptaki her cümlenin, hangi context cümlesine en çok benzediği
   (kullanıcıya "bu cümleyi şuradan aldım" diye gösterilebilir).

Tam bir gradient-tabanlı attribution (örn. SHAP/Integrated Gradients) generative LLM'ler
için hem pahalı hem de yorumlanması zordur; embedding-similarity tabanlı bu yaklaşım
RAG sistemlerinde pratikte en çok kullanılan, ucuz ve anlaşılır alternatiftir.
"""
from __future__ import annotations

import re

import numpy as np

from src.rag.embeddings import embed_passages


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def chunk_level_attribution(answer: str, contexts: list[dict]) -> list[dict]:
    """contexts: [{"text": ..., "doc_title": ...}, ...] -> her birine bir katkı skoru."""
    if not contexts:
        return []
    texts = [answer] + [c["text"] for c in contexts]
    embeddings = np.array(embed_passages(texts))
    answer_emb, context_embs = embeddings[0], embeddings[1:]

    scored = []
    for c, emb in zip(contexts, context_embs):
        scored.append({
            "doc_title": c.get("doc_title"),
            "chunk_id": c.get("chunk_id"),
            "contribution_score": _cosine_sim(answer_emb, emb),
        })
    scored.sort(key=lambda s: s["contribution_score"], reverse=True)
    return scored


def sentence_level_attribution(answer: str, contexts: list[dict]) -> list[dict]:
    """Cevaptaki her cümle için en benzer context cümlesini bulur."""
    answer_sentences = [s for s in re.split(r"(?<=[.!?])\s+", answer.strip()) if len(s) > 3]
    context_sentences = []
    for c in contexts:
        for s in re.split(r"(?<=[.!?])\s+", c["text"]):
            if len(s) > 3:
                context_sentences.append({"text": s, "doc_title": c.get("doc_title")})

    if not answer_sentences or not context_sentences:
        return []

    all_texts = answer_sentences + [cs["text"] for cs in context_sentences]
    embeddings = np.array(embed_passages(all_texts))
    answer_embs = embeddings[: len(answer_sentences)]
    context_embs = embeddings[len(answer_sentences):]

    results = []
    for sent, emb in zip(answer_sentences, answer_embs):
        sims = [_cosine_sim(emb, ce) for ce in context_embs]
        best_idx = int(np.argmax(sims))
        results.append({
            "answer_sentence": sent,
            "best_match": context_sentences[best_idx]["text"],
            "source_doc": context_sentences[best_idx]["doc_title"],
            "similarity": sims[best_idx],
        })
    return results
