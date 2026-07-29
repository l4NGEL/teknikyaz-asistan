"""DPO için tercih (preference) veri seti üretir: chosen vs. rejected cevap çiftleri.

Strateji: "chosen" cevaplar, src/data_prep/instruction_dataset.py'de bağlamdan üretilmiş
(dolayısıyla kaynağa sadık) cevaplardır. "rejected" cevaplar ise AYNI soruyu, modele HİÇ
bağlam vermeden (retrieval olmadan) sorduğumuzda aldığımız serbest üretimdir — bağlamsız
üretim, ezber bilgiye dayandığı için genellikle daha az kesin/daha halüsinasyonludur.

Bu, insan etiketleme gerektirmeyen, ucuz ama gerçek bir "grounded vs ungrounded" sinyali
üretir. Üretimde bunun yerine gerçek kullanıcı geri bildirimi veya insan değerlendirmesi
kullanılır; burada amaç DPO mekaniğini uçtan uca deneyimlemek.
"""
from __future__ import annotations

import json

from src.config import MODEL_CONFIG, PROCESSED_DIR


def _generate_ungrounded_answer(question: str, generator) -> str:
    messages = [
        {"role": "system", "content": "Kısa ve doğrudan Türkçe cevap ver."},
        {"role": "user", "content": question},
    ]
    out = generator(messages, do_sample=True, temperature=0.9, max_new_tokens=150)
    generated = out[0]["generated_text"]
    return generated[-1]["content"] if isinstance(generated, list) else generated


def build_preference_dataset(sft_path=None, max_examples: int | None = None) -> list[dict]:
    from transformers import pipeline

    sft_path = sft_path or (PROCESSED_DIR / "sft_dataset.jsonl")
    generator = pipeline("text-generation", model=MODEL_CONFIG.base_llm, device_map="auto")

    pairs = []
    with open(sft_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_examples and i >= max_examples:
                break
            ex = json.loads(line)
            question = ex["instruction"]
            chosen = ex["output"]
            rejected = _generate_ungrounded_answer(question, generator)

            if rejected.strip() == chosen.strip():
                continue  # ayırt edici sinyal yoksa çifti atla

            pairs.append(
                {
                    "prompt": question,
                    "chosen": chosen,
                    "rejected": rejected,
                    "source_chunk_id": ex.get("source_chunk_id"),
                }
            )
    return pairs


def save_preference_dataset(pairs: list[dict], filename: str = "dpo_pairs.jsonl") -> str:
    out_path = PROCESSED_DIR / filename
    with out_path.open("w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    return str(out_path)


if __name__ == "__main__":
    pairs = build_preference_dataset(max_examples=200)
    path = save_preference_dataset(pairs)
    print(f"{len(pairs)} tercih çifti kaydedildi -> {path}")
