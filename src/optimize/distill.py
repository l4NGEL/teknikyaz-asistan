"""Bilgi damıtma (knowledge distillation): büyük "öğretmen" modelin bilgisini küçük bir
"öğrenci" modele aktarır.

Klasik Hinton damıtma kaybını kullanıyoruz: öğrencinin loss'u, gerçek etiketlerle
çapraz entropi (hard loss) VE öğretmenin yumuşatılmış (temperature ile) olasılık
dağılımına KL-diverjans (soft loss) toplamıdır. Soft loss, öğretmenin "hangi yanlış
sınıfların da makul olduğu" bilgisini taşır — sadece hard label ile eğitmekten daha
zengin bir sinyaldir.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.config import CLASSIFICATION_LABELS, MODELS_DIR
from src.nlp_tasks.classification import TurkishTextDataset

STUDENT_BASE_MODEL = "dbmdz/distilbert-base-turkish-cased"  # öğretmenin ~yarısı boyutunda


def distillation_loss(student_logits, teacher_logits, labels, temperature: float = 2.0, alpha: float = 0.5):
    hard_loss = F.cross_entropy(student_logits, labels)
    soft_loss = F.kl_div(
        F.log_softmax(student_logits / temperature, dim=-1),
        F.softmax(teacher_logits / temperature, dim=-1),
        reduction="batchmean",
    ) * (temperature ** 2)
    return alpha * hard_loss + (1 - alpha) * soft_loss


def distill(
    texts: list[str],
    labels: list[int],
    teacher_dir: str | None = None,
    epochs: int = 3,
    batch_size: int = 8,
    lr: float = 5e-5,
) -> str:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    teacher_dir = teacher_dir or str(MODELS_DIR / "classifier")

    teacher = AutoModelForSequenceClassification.from_pretrained(teacher_dir).to(device).eval()

    student_tokenizer = AutoTokenizer.from_pretrained(STUDENT_BASE_MODEL)
    student = AutoModelForSequenceClassification.from_pretrained(
        STUDENT_BASE_MODEL, num_labels=len(CLASSIFICATION_LABELS)
    ).to(device)

    dataset = TurkishTextDataset(texts, labels, student_tokenizer)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(student.parameters(), lr=lr)

    student.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            labels_batch = batch.pop("labels")

            # Öğretmen ve öğrenci farklı tokenizer'lara sahip olabilir; burada aynı
            # tokenizer ailesini (BERT-tabanlı) kullandığımız için input_ids uyumlu kalır.
            with torch.no_grad():
                teacher_logits = teacher(**batch).logits
            student_logits = student(**batch).logits

            loss = distillation_loss(student_logits, teacher_logits, labels_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"[distill] epoch {epoch + 1}/{epochs} - loss: {total_loss / len(loader):.4f}")

    output_dir = str(MODELS_DIR / "classifier-distilled")
    student.save_pretrained(output_dir)
    student_tokenizer.save_pretrained(output_dir)
    return output_dir
