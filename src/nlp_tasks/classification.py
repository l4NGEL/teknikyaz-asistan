"""Doküman sınıflandırma: chunk'ları CLASSIFICATION_LABELS kategorilerine ayırır.

İki mod sunar:
- zero-shot: Hiç etiketli veri yokken hızlı başlangıç (NLI tabanlı zero-shot pipeline).
- fine-tuned: `train()` ile dbmdz/bert-base-turkish-cased üzerine ince ayar yapılmış
  gerçek bir sınıflandırıcı (ilandaki "metin sınıflandırma" + "PyTorch ile eğitim" maddeleri).

Zero-shot çıktısı, fine-tuning için ilk etiketleri otomatik üretmekte de kullanılabilir
(weak supervision) — küçük veri setlerinde pratik bir teknik.
"""
from __future__ import annotations

import numpy as np
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    pipeline,
)

from src.config import CLASSIFICATION_LABELS, MODEL_CONFIG, MODELS_DIR

_zero_shot_pipe = None


def zero_shot_classify(text: str, labels: list[str] | None = None) -> dict:
    global _zero_shot_pipe
    labels = labels or CLASSIFICATION_LABELS
    if _zero_shot_pipe is None:
        _zero_shot_pipe = pipeline(
            "zero-shot-classification",
            model="joeddav/xlm-roberta-large-xnli",
        )
    result = _zero_shot_pipe(text, candidate_labels=labels, hypothesis_template="Bu metin {} ile ilgilidir.")
    return {"label": result["labels"][0], "scores": dict(zip(result["labels"], result["scores"]))}


class TurkishTextDataset(torch.utils.data.Dataset):
    def __init__(self, texts: list[str], labels: list[int], tokenizer, max_length: int = 256):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length)
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


def train(
    texts: list[str],
    labels: list[int],
    label_names: list[str] | None = None,
    output_dir: str | None = None,
    epochs: int = 3,
) -> str:
    """Etiketli (texts, labels) çiftleriyle sınıflandırıcıyı fine-tune eder ve kaydeder."""
    label_names = label_names or CLASSIFICATION_LABELS
    output_dir = output_dir or str(MODELS_DIR / "classifier")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_CONFIG.classification_base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_CONFIG.classification_base_model, num_labels=len(label_names)
    )

    split = int(len(texts) * 0.85)
    train_ds = TurkishTextDataset(texts[:split], labels[:split], tokenizer)
    eval_ds = TurkishTextDataset(texts[split:], labels[split:], tokenizer) if split < len(texts) else None

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        eval_strategy="epoch" if eval_ds else "no",
        save_strategy="epoch",
        logging_steps=10,
        load_best_model_at_end=bool(eval_ds),
    )

    def compute_metrics(pred):
        preds = np.argmax(pred.predictions, axis=1)
        acc = (preds == pred.label_ids).mean()
        return {"accuracy": float(acc)}

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        compute_metrics=compute_metrics if eval_ds else None,
    )
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    return output_dir


def load_and_classify(text: str, model_dir: str | None = None) -> dict:
    model_dir = model_dir or str(MODELS_DIR / "classifier")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    inputs = tokenizer(text, truncation=True, padding=True, return_tensors="pt", max_length=256)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1).squeeze().tolist()
    idx = int(np.argmax(probs))
    return {"label": CLASSIFICATION_LABELS[idx], "scores": dict(zip(CLASSIFICATION_LABELS, probs))}
