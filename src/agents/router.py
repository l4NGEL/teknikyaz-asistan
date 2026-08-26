"""Soru yönlendirme: greet / retrieve / refuse / clarify.

Çok ajanlı hattın ilk düğümü. LLM çağırmadan, kural tabanlı ve deterministik —
yanlış domain'e GPU yakmamak ve test edilebilir olmak için. Gerekirse ileride
aynı arayüze küçük bir sınıflandırıcı bağlanabilir.
"""
from __future__ import annotations

import re

_PUNCT = re.compile(r"[^\wçğıöşüÇĞİÖŞÜ\s]+", re.IGNORECASE)
_GREETINGS = ("merhaba", "merhabalar", "selam", "selamlar", "hey", "hi", "hello")
_THANKS = ("teşekkür", "teşekkürler", "tesekkur", "tesekkurler", "sağol", "sagol", "thanks", "eyvallah")
_OFF_DOMAIN = (
    "yemek tarifi",
    "hava durumu",
    "futbol skoru",
    "borsa",
    "astroloji",
    "fıkra anlat",
    "şaka yap",
)


def _normalize(question: str) -> str:
    return " ".join(_PUNCT.sub(" ", question or "").casefold().split())


def _is_greeting(normalized: str) -> bool:
    tokens = normalized.split()
    if not tokens:
        return False
    if normalized in _GREETINGS:
        return True
    return len(tokens) <= 3 and tokens[0] in _GREETINGS and all(
        tok in _GREETINGS or tok in ("nasılsın", "nasilsin", "naber", "neler") for tok in tokens[1:]
    )


def _is_thanks(normalized: str) -> bool:
    if not normalized:
        return False
    if any(normalized == word or normalized.startswith(word + " ") for word in _THANKS):
        return True
    tokens = normalized.split()
    return bool(tokens) and tokens[0] in _THANKS and len(tokens) <= 4


def is_thanks(question: str) -> bool:
    return _is_thanks(_normalize(question))


def route(question: str) -> str:
    text = (question or "").strip()
    normalized = _normalize(text)
    if _is_greeting(normalized) or _is_thanks(normalized):
        return "greet"
    if any(phrase in normalized for phrase in _OFF_DOMAIN):
        return "refuse"
    if len(text) < 8:
        return "clarify"
    return "retrieve"
