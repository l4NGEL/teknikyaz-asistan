"""Güven skoru ve kalibrasyon kovaları.

Üretim cevabını tek bir 0–1 skora indirger: NLI groundedness + retrieval skoru,
çelişki cezası. Eşik altında insan onayına yönlendirme `agents.graph` içinde yapılır.

Kovalar (reliability-style): skorun hangi aralıkta olduğunu raporlamak, ham 0.73
sayısını "yüksek / orta / düşük" olarak okunabilir kılar. Tam bir temperature
scaling kalibrasyonu için labeled doğrulama seti gerekir; burada kovalar o
disiplinin iskeletidir.
"""
from __future__ import annotations

import math

from src.config import AGENT_CONFIG

BUCKETS = (
    (0.75, "yüksek"),
    (0.45, "orta"),
    (0.0, "düşük"),
)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def normalize_retrieval_score(score: float | None) -> float:
    """rerank/similarity skorunu 0–1 aralığına çeker. Bilinmiyorsa 0."""
    if score is None:
        return 0.0
    score = float(score)
    if 0.0 <= score <= 1.0:
        return score
    return _clamp01(1.0 / (1.0 + math.exp(-score)))


def retrieval_confidence(contexts: list[dict]) -> float:
    if not contexts:
        return 0.0
    scores = []
    for ctx in contexts:
        raw = ctx.get("rerank_score", ctx.get("similarity", ctx.get("bm25_score")))
        scores.append(normalize_retrieval_score(raw if raw is not None else None))
    return max(scores) if scores else 0.0


def combine_confidence(
    groundedness: float,
    contexts: list[dict],
    n_conflicts: int = 0,
) -> float:
    ret = retrieval_confidence(contexts)
    score = 0.65 * _clamp01(groundedness) + 0.35 * ret
    score -= 0.12 * max(n_conflicts, 0)
    return _clamp01(score)


def confidence_bucket(confidence: float) -> str:
    for threshold, name in BUCKETS:
        if confidence >= threshold:
            return name
    return "düşük"


def should_escalate(
    confidence: float,
    n_conflicts: int = 0,
    threshold: float | None = None,
) -> tuple[bool, str | None]:
    threshold = AGENT_CONFIG.confidence_threshold if threshold is None else threshold
    if n_conflicts > 0:
        return True, f"kaynaklar çelişiyor ({n_conflicts} çelişki)"
    if confidence < threshold:
        return True, f"güven {confidence:.2f} < eşik {threshold:.2f} (kova: {confidence_bucket(confidence)})"
    return False, None


def lexical_groundedness(answer: str, contexts: list[str]) -> float:
    """NLI yokken test/CI için kelime örtüşmesi. Üretimde NLI kullanılır."""
    answer_tokens = {t for t in answer.lower().split() if len(t) > 2}
    if not answer_tokens or not contexts:
        return 0.0
    ctx_tokens = {t for text in contexts for t in text.lower().split() if len(t) > 2}
    if not ctx_tokens:
        return 0.0
    return len(answer_tokens & ctx_tokens) / len(answer_tokens)
