"""Makine çevirisi: TR<->EN, teknik dokümanların çift dilli erişimi için.

Yazılım spesifikasyonlarının/API dokümanlarının çoğu İngilizce kaynaklıdır; TR<->EN
çeviri katmanı hem korpus zenginleştirme (İngilizce spesifikasyonu Türkçeye çevirip
taslak notlara katma) hem de kullanıcıya iki dilde dokümantasyon sunma senaryolarında
kullanılabilir.
"""
from __future__ import annotations

from transformers import pipeline

from src.config import MODEL_CONFIG

_pipes: dict[str, object] = {}


def _get_pipe(direction: str):
    if direction not in _pipes:
        model_name = MODEL_CONFIG.mt_tr_en_model if direction == "tr-en" else MODEL_CONFIG.mt_en_tr_model
        _pipes[direction] = pipeline("translation", model=model_name)
    return _pipes[direction]


def translate(text: str, direction: str = "tr-en") -> str:
    if direction not in ("tr-en", "en-tr"):
        raise ValueError("direction 'tr-en' ya da 'en-tr' olmalı")
    pipe = _get_pipe(direction)
    return pipe(text)[0]["translation_text"]


if __name__ == "__main__":
    print(translate("Bu fonksiyon, kullanıcı e-postası zaten kayıtlıysa bir hata fırlatır.", "tr-en"))
