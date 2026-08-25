"""Soru yönlendirme: retrieve / refuse / clarify.

Çok ajanlı hattın ilk düğümü. LLM çağırmadan, kural tabanlı ve deterministik —
yanlış domain'e GPU yakmamak ve test edilebilir olmak için. Gerekirse ileride
aynı arayüze küçük bir sınıflandırıcı bağlanabilir.
"""
from __future__ import annotations

_GREETINGS = ("merhaba", "selam", "hey", "hi", "hello")
_OFF_DOMAIN = (
    "yemek tarifi",
    "hava durumu",
    "futbol skoru",
    "borsa",
    "astroloji",
    "fıkra anlat",
    "şaka yap",
)


def route(question: str) -> str:
    text = (question or "").strip()
    lowered = text.lower()
    tokens = lowered.split()
    if lowered in _GREETINGS or (len(tokens) <= 2 and tokens and tokens[0] in _GREETINGS):
        return "refuse"
    if any(phrase in lowered for phrase in _OFF_DOMAIN):
        return "refuse"
    if len(text) < 8:
        return "clarify"
    return "retrieve"
