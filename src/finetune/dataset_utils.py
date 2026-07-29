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
    """(instruction, input, output) -> chat mesajları. TRL'in SFTTrainer'ı bu formatı bekler."""
    user_content = example["instruction"]
    if example.get("input"):
        user_content = f"{example['input']}\n\nSoru: {example['instruction']}"
    return {
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": example["output"]},
        ]
    }


def build_hf_dataset(path=None, val_ratio: float = 0.1) -> tuple[Dataset, Dataset]:
    examples = load_sft_examples(path)
    chat_examples = [to_chat_format(e) for e in examples]

    ds = Dataset.from_list(chat_examples)
    split = ds.train_test_split(test_size=val_ratio, seed=42)
    return split["train"], split["test"]
