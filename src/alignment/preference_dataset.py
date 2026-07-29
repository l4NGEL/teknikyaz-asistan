"""DPO için tercih (preference) veri seti üretir: chosen vs. rejected doküman çiftleri.

Strateji: "chosen" dokümanlar, src/data_prep/instruction_dataset.py'de zaten hedef
kalitede üretilmiş (iyi yapılandırılmış, tutarlı) çıktılardır. "rejected" dokümanlar ise
AYNI taslak notlar verildiğinde, modele hiçbir şablon/stil bağlamı vermeden (retrieval
olmadan) ürettirdiğimiz serbest üretimdir — bağlamsız üretim genellikle daha dağınık,
tutarsız yapıda veya notlarda olmayan ayrıntılar "uyduran" bir metin olur.

Bu, insan etiketleme gerektirmeyen, ucuz ama gerçek bir "iyi yapılandırılmış vs. serbest/
tutarsız" sinyali üretir. Üretimde bunun yerine gerçek kullanıcı geri bildirimi veya insan
değerlendirmesi kullanılır; burada amaç DPO mekaniğini uçtan uca deneyimlemek.
"""
from __future__ import annotations

import json

from src.config import MODEL_CONFIG, PROCESSED_DIR


def _generate_ungrounded_answer(prompt_text: str, generator) -> str:
    messages = [
        {"role": "system", "content": "Kısa ve doğrudan Türkçe yanıt ver, şablon veya stil kılavuzu kullanma."},
        {"role": "user", "content": prompt_text},
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
            # SFT örneklerinde `instruction` sabittir (bkz. instruction_dataset.py); asıl
            # değişen içerik `input`teki (taslak notlar) kısımdır, bu yüzden prompt'u
            # ikisini birleştirerek kuruyoruz — tıpkı fine-tuning'de kullanılan format gibi.
            prompt_text = f"{ex['instruction']}\n\n{ex['input']}"
            chosen = ex["output"]
            rejected = _generate_ungrounded_answer(prompt_text, generator)

            if rejected.strip() == chosen.strip():
                continue  # ayırt edici sinyal yoksa çifti atla

            pairs.append(
                {
                    "prompt": prompt_text,
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
