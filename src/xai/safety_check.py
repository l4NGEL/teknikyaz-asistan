"""Sorumlu yapay zekâ katmanı: yanıtı kullanıcıya döndürmeden önce basit güvenlik/önyargı
kontrolünden geçirir.

İki katman:
1. Model tabanlı: Türkçe bir "offensive/toxic language" sınıflandırıcısı.
2. Kural tabanlı: domain'e özgü hassas terimler (örn. can kaybı/casualty rakamları gibi
   yanlış aktarılırsa zarar verebilecek ifadeler) için basit bir anahtar kelime taraması.

Model tabanlı kontrol tek başına yeterli değildir (yanlış negatif/pozitif verebilir);
kural tabanlı katman, domain'e özgü riskleri modelin kaçırdığı yerlerde yakalamak için
ikinci bir güvenlik ağı görevi görür — üretim sistemlerinde "defense in depth" yaklaşımı.
"""
from __future__ import annotations

from transformers import pipeline

from src.config import MODEL_CONFIG

_offensive_pipe = None

SENSITIVE_KEYWORDS = [
    "kesin sayıda ölü", "kesin kayıp rakamı", "resmi olmayan istihbarat",
]


def _get_pipe():
    global _offensive_pipe
    if _offensive_pipe is None:
        _offensive_pipe = pipeline("text-classification", model=MODEL_CONFIG.offensive_model)
    return _offensive_pipe


def model_based_check(text: str) -> dict:
    result = _get_pipe()(text)[0]
    is_offensive = result["label"].upper() in ("OFFENSIVE", "LABEL_1", "1")
    return {"is_offensive": is_offensive, "label": result["label"], "score": float(result["score"])}


def keyword_based_check(text: str) -> list[str]:
    text_lower = text.lower()
    return [kw for kw in SENSITIVE_KEYWORDS if kw in text_lower]


def safety_check(text: str) -> dict:
    model_result = model_based_check(text)
    keyword_hits = keyword_based_check(text)

    is_safe = (not model_result["is_offensive"]) and (not keyword_hits)
    return {
        "is_safe": is_safe,
        "model_check": model_result,
        "keyword_hits": keyword_hits,
    }
