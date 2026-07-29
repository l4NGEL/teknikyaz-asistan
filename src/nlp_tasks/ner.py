"""Varlık tanıma (NER): hazır Türkçe NER modeli + teknik-dokümantasyona özgü regex desenleri.

Genel amaçlı NER modelleri kişi/kurum/yer varlıklarını yakalar ama teknik dokümantasyonun
kendine özgü "varlıkları" (fonksiyon adı, CLI bayrağı, sürüm numarası, ortam değişkeni,
dosya yolu) bunların dışında kalır — çünkü bunlar kapalı bir isim listesi (gazetteer)
değil, düzenli (regex ile yakalanabilir) kalıplardır. Bu yüzden burada model çıktısını,
sabit bir kelime listesi yerine desen-tabanlı bir çıkarımla zenginleştiriyoruz (hibrit NER).
"""
from __future__ import annotations

import re

from transformers import pipeline

from src.config import MODEL_CONFIG

_ner_pipe = None

# Teknik dokümantasyonda sık geçen, regex ile güvenilir şekilde yakalanabilen kalıplar.
TECHNICAL_PATTERNS: dict[str, str] = {
    "KOD_PARCASI": r"`[^`\n]+`",
    "CLI_BAYRAGI": r"(?<![\w-])--[a-zA-Z][\w-]*|(?<![\w-])-[a-zA-Z](?![\w-])",
    "SURUM_NUMARASI": r"\bv?\d+\.\d+(?:\.\d+)?\b",
    "ORTAM_DEGISKENI": r"\b[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+\b",
    "DOSYA_YOLU": r"(?:[.~]?/[\w./-]+)|(?:[A-Za-z]:\\[\w\\.-]+)",
    "FONKSIYON_CAGRISI": r"\b[a-z_][a-z0-9_]*\([^)\n]*\)",
}


def _get_pipe():
    global _ner_pipe
    if _ner_pipe is None:
        _ner_pipe = pipeline(
            "ner",
            model=MODEL_CONFIG.ner_model,
            aggregation_strategy="simple",
        )
    return _ner_pipe


def _pattern_matches(text: str) -> list[dict]:
    matches = []
    for label, pattern in TECHNICAL_PATTERNS.items():
        for m in re.finditer(pattern, text):
            matches.append(
                {
                    "entity_group": label,
                    "word": m.group(0),
                    "start": m.start(),
                    "end": m.end(),
                    "score": 1.0,
                    "source": "pattern",
                }
            )
    return matches


def _remove_overlaps(entities: list[dict]) -> list[dict]:
    entities = sorted(entities, key=lambda e: (e["start"], -(e["end"] - e["start"])))
    result = []
    last_end = -1
    for e in entities:
        if e["start"] >= last_end:
            result.append(e)
            last_end = e["end"]
    return result


def extract_entities(text: str) -> list[dict]:
    model_entities = [
        {**e, "source": "model", "score": float(e.get("score", 0))} for e in _get_pipe()(text)
    ]
    pattern_entities = _pattern_matches(text)
    # Regex desenlerine öncelik veriyoruz çünkü tam eşleşme, domain'e özgü ve kesindir.
    combined = _remove_overlaps(pattern_entities + model_entities)
    return combined


if __name__ == "__main__":
    sample = (
        "`kullanici_olustur(ad, e_posta)` fonksiyonunu çağırmadan önce ORTAM_DEGISKENI "
        "olan DATABASE_URL'yi ayarlayın. Sürüm v2.3.0'dan itibaren --force bayrağı "
        "eklendi, yapılandırma dosyası /etc/proje/config.yaml içinde bulunur."
    )
    for ent in extract_entities(sample):
        print(ent)
