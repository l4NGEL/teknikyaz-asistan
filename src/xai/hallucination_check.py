"""Halüsinasyon tespiti: üretilen cevabın, verilen bağlam(lar) tarafından desteklenip
desteklenmediğini bir NLI (Natural Language Inference) modeliyle ölçer.

Yöntem: cevabı cümlelere böl, her cümle için "bağlam bu cümleyi ENTAIL (mantıksal olarak
destekler) mi?" sorusunu NLI modeline sor. Cümlelerin ENTAILMENT olasılıklarının ortalaması
"groundedness" (kaynağa dayanma) skorunu verir — 1.0'a yakınsa cevap büyük ölçüde
bağlamdan destekleniyor demektir, düşükse halüsinasyon riski yüksektir.

Bu, tam bir halüsinasyon tespit sistemi değildir (üretim sistemlerinde insan
değerlendirmesi + daha güçlü modellerle çapraz kontrol de eklenir) ama ilandaki
"halüsinasyon azaltma" maddesine somut, çalışan bir uygulama sağlar.
"""
from __future__ import annotations

import re

from transformers import pipeline

from src.config import MODEL_CONFIG

_nli_pipe = None


def _get_pipe():
    global _nli_pipe
    if _nli_pipe is None:
        _nli_pipe = pipeline("text-classification", model=MODEL_CONFIG.nli_model, top_k=None)
    return _nli_pipe


def _split_sentences(text: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s for s in sentences if len(s) > 3]


def _entailment_prob(premise: str, hypothesis: str) -> float:
    pipe = _get_pipe()
    # mDeBERTa-mnli-xnli çıktı etiketleri: entailment / neutral / contradiction
    scores = pipe(f"{premise}</s></s>{hypothesis}")[0]
    for s in scores:
        if s["label"].lower().startswith("entail"):
            return float(s["score"])
    return 0.0


def groundedness_score(answer: str, contexts: list[str]) -> float:
    """0.0 (tamamen desteksiz) - 1.0 (tamamen bağlamdan destekli) arası skor döndürür."""
    sentences = _split_sentences(answer)
    if not sentences or not contexts:
        return 0.0

    combined_context = " ".join(contexts)
    sentence_scores = []
    for sentence in sentences:
        best = max(_entailment_prob(combined_context, sentence), 0.0)
        sentence_scores.append(best)
    return sum(sentence_scores) / len(sentence_scores)


def flag_unsupported_sentences(answer: str, contexts: list[str], threshold: float = 0.4) -> list[dict]:
    """Bağlam tarafından yeterince desteklenmeyen (entailment < threshold) cümleleri işaretler."""
    combined_context = " ".join(contexts)
    flagged = []
    for sentence in _split_sentences(answer):
        score = _entailment_prob(combined_context, sentence)
        if score < threshold:
            flagged.append({"sentence": sentence, "entailment_score": score})
    return flagged
