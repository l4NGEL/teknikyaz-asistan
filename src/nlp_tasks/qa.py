"""Soru-cevap: extractive (bağlamdan span çıkarır) + generative (RAG üzerinden, bkz. src/rag).

Extractive QA'nın avantajı: cevap her zaman kaynak metinden birebir alınır, halüsinasyon
imkansızdır — ama sadece kaynakta *kelimesi kelimesine* geçen cevaplar için çalışır.
Generative QA (rag_pipeline.answer) daha esnektir ama XAI modülündeki groundedness
kontrolünden geçmelidir.
"""
from __future__ import annotations

from transformers import pipeline

from src.config import MODEL_CONFIG

_qa_pipe = None


def _get_pipe():
    global _qa_pipe
    if _qa_pipe is None:
        _qa_pipe = pipeline("question-answering", model=MODEL_CONFIG.qa_model)
    return _qa_pipe


def extractive_answer(question: str, context: str) -> dict:
    pipe = _get_pipe()
    result = pipe(question=question, context=context)
    return {
        "answer": result["answer"],
        "score": float(result["score"]),
        "start": result["start"],
        "end": result["end"],
    }


if __name__ == "__main__":
    ctx = "kullanici_olustur fonksiyonu, e_posta parametresi zaten kayıtlıysa DuplicateEmailError fırlatır."
    print(extractive_answer("kullanici_olustur hangi hatayı fırlatır?", ctx))
