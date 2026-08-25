"""SFT veri setini (instruction/input/output) sohbet şablonuna çevirip train/val böler."""
from __future__ import annotations

import json

from datasets import Dataset

from src.config import PROCESSED_DIR


def load_sft_examples(path=None) -> list[dict]:
    path = path or (PROCESSED_DIR / "sft_dataset.jsonl")
    examples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            examples.append(json.loads(line))
    return examples


def to_chat_format(example: dict) -> dict:
    """(instruction, input, output[, contexts]) -> chat mesajları. TRL'in SFTTrainer'ı bu formatı bekler.

    `rag_pipeline.build_prompt`'un TA KENDİSİNİ çağırıyoruz (aynı sistem promptu, aynı
    "Bağlam:/Soru:" biçimi) ki eğitim ve çıkarım promptu asla birbirinden ayrışmasın --
    bunun ayrışması, modelin gerçek RAG bağlamını hiç görmeden eğitilmesine (ve çıkarımda
    bağlamı sadece kopyalamasına) yol açan asıl sorundu.
    """
    from src.rag.rag_pipeline import SYSTEM_PROMPT, build_prompt

    question = example["instruction"]
    if example.get("input"):
        question = f"{example['instruction']}\n\n{example['input']}"

    contexts = example.get("contexts") or []
    if contexts:
        messages = build_prompt(question, contexts)
    else:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
    messages.append({"role": "assistant", "content": example["output"]})
    return {"messages": messages}


def build_hf_dataset(path=None, val_ratio: float = 0.1) -> tuple[Dataset, Dataset]:
    examples = load_sft_examples(path)
    chat_examples = [to_chat_format(e) for e in examples]

    ds = Dataset.from_list(chat_examples)
    split = ds.train_test_split(test_size=val_ratio, seed=42)
    return split["train"], split["test"]
