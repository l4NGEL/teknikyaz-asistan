"""Varlık tanıma (NER): hazır Türkçe NER modeli + domain'e özgü gazetteer birleşimi.

Genel amaçlı NER modelleri "Bayraktar TB2" gibi platform isimlerini ya PERSON/ORG
olarak yanlış etiketler ya da hiç yakalamaz. Bunu düzeltmek için basit ama etkili bir
teknik kullanıyoruz: model çıktısını, config.py'deki DOMAIN_GAZETTEER ile post-process
ederek zenginleştiriyoruz (gerçek endüstride "hibrit NER" olarak bilinir).
"""
from __future__ import annotations

import re

from transformers import pipeline

from src.config import DOMAIN_GAZETTEER, MODEL_CONFIG

_ner_pipe = None


def _get_pipe():
    global _ner_pipe
    if _ner_pipe is None:
        _ner_pipe = pipeline(
            "ner",
            model=MODEL_CONFIG.ner_model,
            aggregation_strategy="simple",
        )
    return _ner_pipe


def _gazetteer_matches(text: str) -> list[dict]:
    matches = []
    for label, terms in DOMAIN_GAZETTEER.items():
        for term in sorted(terms, key=len, reverse=True):  # uzun terimler önce (örtüşmeyi azaltır)
            for m in re.finditer(re.escape(term), text):
                matches.append(
                    {
                        "entity_group": label,
                        "word": term,
                        "start": m.start(),
                        "end": m.end(),
                        "score": 1.0,
                        "source": "gazetteer",
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
    gazetteer_entities = _gazetteer_matches(text)
    # Gazetteer'a öncelik veriyoruz çünkü domain'e özgü ve kesin (regex tam eşleşme).
    combined = _remove_overlaps(gazetteer_entities + model_entities)
    return combined


if __name__ == "__main__":
    sample = "Baykar tarafından geliştirilen Bayraktar TB2, ASELSAN'ın ürettiği elektro-optik sistemlerle donatılmıştır."
    for ent in extract_entities(sample):
        print(ent)
