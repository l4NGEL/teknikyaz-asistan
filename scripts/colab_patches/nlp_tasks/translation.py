"""Makine çevirisi: TR<->EN, teknik dokümanların çift dilli erişimi için.

Yazılım spesifikasyonlarının/API dokümanlarının çoğu İngilizce kaynaklıdır; TR<->EN
çeviri katmanı hem korpus zenginleştirme (İngilizce spesifikasyonu Türkçeye çevirip
taslak notlara katma) hem de kullanıcıya iki dilde dokümantasyon sunma senaryolarında
kullanılabilir.
"""
from __future__ import annotations

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from src.config import MODEL_CONFIG

_models: dict[str, AutoModelForSeq2SeqLM] = {}
_tokenizers: dict[str, AutoTokenizer] = {}


def _get_translation_model(direction: str):
    if direction not in _models:
        model_name = MODEL_CONFIG.mt_tr_en_model if direction == "tr-en" else MODEL_CONFIG.mt_en_tr_model
        _tokenizers[direction] = AutoTokenizer.from_pretrained(model_name)
        _models[direction] = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _models[direction].to(device)
    return _models[direction], _tokenizers[direction]


def translate(text: str, direction: str = "tr-en") -> str:
    if direction not in ("tr-en", "en-tr"):
        raise ValueError("direction 'tr-en' ya da 'en-tr' olmalı")
    model, tokenizer = _get_translation_model(direction)
    device = next(model.parameters()).device
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    generated = model.generate(**inputs, max_length=512)
    return tokenizer.decode(generated[0], skip_special_tokens=True)


if __name__ == "__main__":
    print(translate("Bu fonksiyon, kullanıcı e-postası zaten kayıtlıysa bir hata fırlatır.", "tr-en"))
