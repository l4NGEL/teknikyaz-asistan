"""Chunk'lardan instruction-tuning (SFT) veri seti üretir: "taslak notlar -> düzgün doküman".

Yaklaşım, "self-instruct" tekniğinin TERS yönde uygulanmış hali: elimizde zaten hedef
kalitede, iyi yazılmış Türkçe teknik doküman parçaları (chunk'lar) var. Bir LLM'e bu
parçayı verip "bir kullanıcının bunu isterken yazacağı kaba taslak notları/madde
işaretlerini üret" diyoruz. Böylece (kaba taslak -> düzgün doküman) çiftleri elde ediyoruz
— bu, fine-tuning aşamasında modele "kullanıcının dağınık notlarını al, yapılandırılmış,
iyi yazılmış Türkçe teknik dokümana dönüştür" davranışını öğretecek veri setidir.

GPU/LLM erişimi yoksa `use_llm=False` ile basit bir sezgisel (heuristic) "taslaklaştırma"
moduna düşer — cümleleri kısaltıp madde işaretine çevirir. Bu, CPU'da da uçtan uca test
imkanı sağlar.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

from src.config import MODEL_CONFIG, PROCESSED_DIR

GENERATION_PROMPT = """Aşağıda iyi yazılmış bir Türkçe teknik dokümantasyon parçası var.
Bir kullanıcının bu dokümanı yazdırmak için sana vereceği KABA, dağınık, madde işaretli
taslak notları üret. Notlar kısa ve eksik cümleler içerebilir, tam cümleler kurma.

Doküman:
\"\"\"{context}\"\"\"

Şu JSON formatında yanıt ver (başka hiçbir şey yazma):
{{"rough_notes": "..."}}
"""


@dataclass
class InstructionExample:
    instruction: str
    input: str
    output: str
    source_chunk_id: str
    source_doc_title: str

    def to_dict(self) -> dict:
        return {
            "instruction": self.instruction,
            "input": self.input,
            "output": self.output,
            "source_chunk_id": self.source_chunk_id,
            "source_doc_title": self.source_doc_title,
        }


def _parse_llm_json(raw: str) -> dict | None:
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def _heuristic_rough_notes(chunk_text: str, max_sentences: int = 4) -> str | None:
    """LLM olmadan kaba bir 'taslaklaştırma': cümleleri kısaltıp madde işaretine çevirir."""
    sentences = re.split(r"(?<=[.!?])\s+", chunk_text.strip())
    sentences = [s for s in sentences if len(s) > 15][:max_sentences]
    if not sentences:
        return None

    bullets = []
    for s in sentences:
        words = s.rstrip(".!?").split()
        short = " ".join(words[:6])  # ilk birkaç kelimeyi al, "not" havası ver
        bullets.append(f"- {short}")
    return "\n".join(bullets)


FIXED_INSTRUCTION = (
    "Aşağıdaki kaba taslak notları, iyi yapılandırılmış, akıcı bir Türkçe teknik "
    "dokümana dönüştür."
)


def build_sft_dataset(chunks_path=None, use_llm: bool = True, max_examples: int | None = None) -> list[InstructionExample]:
    chunks_path = chunks_path or (PROCESSED_DIR / "chunks.jsonl")
    examples: list[InstructionExample] = []

    generator = None
    if use_llm:
        from transformers import pipeline
        generator = pipeline(
            "text-generation",
            model=MODEL_CONFIG.base_llm,
            device_map="auto",
            max_new_tokens=200,
        )

    with open(chunks_path, encoding="utf-8") as f:
        for line in f:
            chunk = json.loads(line)
            if max_examples and len(examples) >= max_examples:
                break

            rough_notes = None
            if generator is not None:
                prompt = GENERATION_PROMPT.format(context=chunk["text"])
                messages = [{"role": "user", "content": prompt}]
                out = generator(messages, do_sample=True, temperature=0.7)
                generated = out[0]["generated_text"]
                raw_text = generated[-1]["content"] if isinstance(generated, list) else generated
                parsed = _parse_llm_json(raw_text)
                if parsed and "rough_notes" in parsed:
                    rough_notes = parsed["rough_notes"]

            if rough_notes is None:
                rough_notes = _heuristic_rough_notes(chunk["text"])
            if rough_notes is None:
                continue

            examples.append(
                InstructionExample(
                    instruction=FIXED_INSTRUCTION,
                    input=f"Taslak notlar:\n{rough_notes}",
                    output=chunk["text"],
                    source_chunk_id=chunk["chunk_id"],
                    source_doc_title=chunk["doc_title"],
                )
            )
    return examples


def save_sft_dataset(examples: list[InstructionExample], filename: str = "sft_dataset.jsonl") -> str:
    out_path = PROCESSED_DIR / filename
    with out_path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex.to_dict(), ensure_ascii=False) + "\n")
    return str(out_path)


if __name__ == "__main__":
    # Varsayılan: hızlı deneme için LLM'siz sezgisel mod. Colab'da gerçek üretim için
    # `build_sft_dataset(use_llm=True)` çağırın.
    dataset = build_sft_dataset(use_llm=False)
    path = save_sft_dataset(dataset)
    print(f"{len(dataset)} SFT örneği kaydedildi -> {path}")
