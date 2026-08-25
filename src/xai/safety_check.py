"""Sorumlu yapay zekâ katmanı: yanıtı kullanıcıya döndürmeden önce basit güvenlik/önyargı
kontrolünden geçirir.

İki katman:
1. Model tabanlı: Türkçe bir "offensive/toxic language" sınıflandırıcısı.
2. Kural tabanlı: bir dokümantasyon yazım asistanına özgü asıl risk — modelin, örnek kod
   bloklarına gerçek görünümlü kimlik bilgisi (API anahtarı, şifre, token) "uydurması"
   ya da kullanıcının verdiği notlardaki gerçek bir sırrı olduğu gibi dokümana kopyalaması.
   Bunu regex tabanlı bir sır/kimlik-bilgisi deseni taramasıyla yakalıyoruz.

Model tabanlı kontrol tek başına yeterli değildir (yanlış negatif/pozitif verebilir);
kural tabanlı katman, domain'e özgü riskleri modelin kaçırdığı yerlerde yakalamak için
ikinci bir güvenlik ağı görevi görür — üretim sistemlerinde "defense in depth" yaklaşımı.
"""
from __future__ import annotations

import re

from transformers import pipeline

from src.config import MODEL_CONFIG

_offensive_pipe = None

# Örnek kod/komut bloklarında sızabilecek, gerçek görünümlü kimlik bilgisi kalıpları.
SECRET_PATTERNS: dict[str, str] = {
    "OLASI_SIFRE": r"(?i)\bpassword\s*=\s*['\"]?[^\s'\"]{4,}",
    "OLASI_API_ANAHTARI": r"(?i)\b(api[_-]?key|secret[_-]?key|token)\s*=\s*['\"]?[A-Za-z0-9_\-]{8,}",
    "AWS_ERISIM_ANAHTARI": r"\bAKIA[0-9A-Z]{16}\b",
    "OZEL_ANAHTAR_BASLIGI": r"-----BEGIN (RSA |EC )?PRIVATE KEY-----",
}


def _get_pipe():
    global _offensive_pipe
    if _offensive_pipe is None:
        _offensive_pipe = pipeline("text-classification", model=MODEL_CONFIG.offensive_model)
    return _offensive_pipe


def model_based_check(text: str) -> dict:
    result = _get_pipe()(text)[0]
    # Bu model ikili (offensive/not) degil -- 5 sinifli: INSULT, OTHER, PROFANITY,
    # RACIST, SEXIST (bkz. model config.json id2label). "OTHER" saldirgan olmayan
    # genel kategori; digger 4 etiketten HERHANGI biri saldirgan icerik demektir.
    # (Onceki hali "OFFENSIVE"/"LABEL_1"/"1" ariyordu -- bu model hicbir zaman bu
    # etiketleri uretmedigi icin is_offensive HER ZAMAN False donuyordu, model
    # tabanli kontrol sessizce hicbir sey yakalamiyordu.)
    is_offensive = result["label"].upper() != "OTHER"
    return {"is_offensive": is_offensive, "label": result["label"], "score": float(result["score"])}


def secret_leak_check(text: str) -> list[dict]:
    hits = []
    for label, pattern in SECRET_PATTERNS.items():
        for m in re.finditer(pattern, text):
            hits.append({"pattern": label, "match": m.group(0)})
    return hits


def safety_check(text: str) -> dict:
    model_result = model_based_check(text)
    secret_hits = secret_leak_check(text)

    is_safe = (not model_result["is_offensive"]) and (not secret_hits)
    return {
        "is_safe": is_safe,
        "model_check": model_result,
        "secret_hits": secret_hits,
    }
