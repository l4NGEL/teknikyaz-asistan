"""GPU / indeks yokken mülakat demosu için sabit turlar.

Aynı router kurallarını kullanır (refuse / clarify / retrieve). Retrieve turları
korpustaki gerçek şablon başlıklarına yaslanır; uydurma belge yok.
"""
from __future__ import annotations

from src.agents.router import route

DEMO_EXAMPLES = (
    "Bir README'de hangi bölümler olmalı?",
    "API dokümantasyonu nasıl yapılandırılır?",
    "Drone sürü zekası belgesi nasıl yazılır?",
    "hava durumu yarın nasıl",
)


def _turn(
    *,
    answer: str,
    sources: list[dict],
    confidence: float,
    confidence_bucket: str,
    agent_trace: list[str],
    needs_human_review: bool = False,
    review_reason: str | None = None,
    draft_answer: str | None = None,
) -> dict:
    return {
        "answer": answer,
        "sources": sources,
        "confidence": confidence,
        "confidence_bucket": confidence_bucket,
        "needs_human_review": needs_human_review,
        "review_reason": review_reason,
        "draft_answer": draft_answer,
        "agent_trace": agent_trace,
        "demo": True,
    }


README = _turn(
    answer=(
        "İyi bir README en az şunları kapsar: proje özeti, özellikler, kurulum, "
        "kullanım örnekleri, yapılandırma ve lisans. İsteğe bağlı olarak katkı "
        "rehberi ve SSS eklenir. Başlıklar tarayıcının içindekiler listesini "
        "üretecek kadar tutarlı tutulmalıdır."
    ),
    sources=[{"title": "README Şablonu", "url": "https://example.com/readme-sablonu"}],
    confidence=0.86,
    confidence_bucket="yüksek",
    agent_trace=["router:retrieve", "retrieve:5", "writer", "critic:g=0.84,conflicts=0", "consensus"],
)

API = _turn(
    answer=(
        "API dokümantasyonu OpenAPI şemasıyla başlar; her endpoint için yöntem, "
        "yol, istek/yanıt örnekleri ve hata kodları yazılır. Kimlik doğrulama ve "
        "hız sınırı ayrı bölümlerde toplanır ki istemci üreticisi şemayı tek "
        "kaynak olarak kullanabilsin."
    ),
    sources=[{"title": "API Dokümantasyonu Şablonu", "url": "https://example.com/api-sablonu"}],
    confidence=0.81,
    confidence_bucket="yüksek",
    agent_trace=["router:retrieve", "retrieve:5", "writer", "critic:g=0.79,conflicts=0", "consensus"],
)

HITL = _turn(
    answer=(
        "Bu yanıtın güven skoru düşük olduğu için insan onayına yönlendirildi. "
        "Gerekçe: güven 0.28 < eşik 0.45 (kova: düşük)"
    ),
    sources=[],
    confidence=0.28,
    confidence_bucket="düşük",
    needs_human_review=True,
    review_reason="güven 0.28 < eşik 0.45 (kova: düşük)",
    draft_answer=(
        "Korpus bu başlığı içermiyor. Taslak: sürü zekası belgesi genelde hedef, "
        "iletişim modeli ve güvenlik sınırlarını anlatır; bu metin şablonlara "
        "dayanmadığı için yayımlanmamalı."
    ),
    agent_trace=["router:retrieve", "retrieve:0", "writer:empty", "critic:g=0.00,conflicts=0", "consensus:human"],
)

REFUSE = _turn(
    answer=(
        "Bu soru teknik dokümantasyon kapsamının dışında. README, API, kurulum "
        "veya mimari belgeler hakkında sorun."
    ),
    sources=[],
    confidence=1.0,
    confidence_bucket="yüksek",
    agent_trace=["router:refuse", "refuse"],
)

CLARIFY = _turn(
    answer="Soruyu biraz açabilir misiniz? Hangi belge türünü (README, API, kurulum, mimari) kastediyorsunuz?",
    sources=[],
    confidence=1.0,
    confidence_bucket="yüksek",
    agent_trace=["router:clarify", "clarify"],
)

_RETRIEVE_CASES = (
    (("readme", "lisans", "kurulum", "bölüm"), README),
    (("api", "openapi", "endpoint"), API),
)


def lookup_demo(question: str) -> dict:
    decision = route(question)
    if decision == "refuse":
        return dict(REFUSE)
    if decision == "clarify":
        return dict(CLARIFY)

    lowered = (question or "").casefold()
    best: tuple[int, dict] | None = None
    for keywords, payload in _RETRIEVE_CASES:
        hits = sum(1 for key in keywords if key in lowered)
        if hits and (best is None or hits > best[0]):
            best = (hits, payload)
    if best:
        return dict(best[1])
    return dict(HITL)
