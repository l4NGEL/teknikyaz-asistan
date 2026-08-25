"""Doküman sınıflandırma: chunk'ları CLASSIFICATION_LABELS kategorilerine ayırır.

İki mod sunar:
- zero-shot: Hiç etiketli veri yokken hızlı başlangıç (NLI tabanlı zero-shot pipeline).
- fine-tuned: `train()` ile dbmdz/bert-base-turkish-cased üzerine ince ayar yapılmış
  gerçek bir sınıflandırıcı (ilandaki "metin sınıflandırma" + "PyTorch ile eğitim" maddeleri).

Zero-shot çıktısı, fine-tuning için ilk etiketleri otomatik üretmekte de kullanılabilir
(weak supervision) — küçük veri setlerinde pratik bir teknik.
"""
from __future__ import annotations

import re

import numpy as np
import torch
from sklearn.metrics import classification_report as _sk_classification_report
from sklearn.metrics import precision_recall_fscore_support
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    pipeline,
)

from src.config import CLASSIFICATION_LABELS, MODEL_CONFIG, MODELS_DIR


def label_for_title(doc_title: str) -> str:
    """Doküman başlığını CLASSIFICATION_LABELS kategorilerinden birine eşler.

    Önek eşleştirmesi kullanır (tam eşleştirme değil): örn. "API Referansı: X" ve
    "API Dokümantasyonu Şablonu" ikisi de "API Dokümantasyonu"ya eşlenir. Bu fonksiyon
    src/'de tek bir yerde tutulur ve buradan import edilir -- daha önce birden fazla
    notebook hücresinde ayrı ayrı (ve tutarsız biçimde, biri düzeltilip diğeri
    unutularak) kopyalanmıştı, bu da aynı etiketleme hatasının farklı yerlerde
    tekrar tekrar ortaya çıkmasına yol açmıştı.
    """
    prefixes = [
        ("API Referansı", "API Dokümantasyonu"),
        ("README", "README"),
        ("API Dokümantasyonu", "API Dokümantasyonu"),
        ("Kurulum Kılavuzu", "Kurulum Kılavuzu"),
        ("Kullanıcı Kılavuzu", "Kullanıcı Kılavuzu"),
        ("SSS", "SSS"),
        ("Mimari Doküman", "Mimari Doküman"),
        ("Sürüm Notları", "Sürüm Notları"),
    ]
    for prefix, label in prefixes:
        if doc_title.startswith(prefix):
            return label
    return "Diğer"


def _slugify_label(label: str) -> str:
    """Metrik ismi olarak kullanilabilecek ASCII-guvenli kisa ad (MLflow dosya
    tabanli depoda metrik adi dogrudan dosya adi oluyor; Turkce/ozel karakterler
    ve bosluklar sorun cikarabiliyor)."""
    table = str.maketrans("çğıöşüİÇĞIÖŞÜ", "cgiosuicgiosu")
    slug = label.translate(table).lower()
    return re.sub(r"[^a-z0-9]+", "_", slug).strip("_")

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
        report_to=["mlflow"],  # report_to belirtilmezse Trainer kurulu HER tracker'i (orn.
        # wandb) otomatik etkinlestiriyor ve ilk kullanimda interaktif bir giris/hesap
        # sorusuyla notebook hücresini bloke ediyor -- diger egitim fonksiyonlarindaki
        # (qlora_train.py, dpo_train.py) gibi sadece mlflow'a sabitliyoruz.
    )

    def compute_metrics(pred):
        preds = np.argmax(pred.predictions, axis=1)
        gold = pred.label_ids
        acc = (preds == gold).mean()

        # Kucuk eval setlerinde tek bir "accuracy" sayisi yaniltici olabiliyor (orn.
        # coğunluk sinifini her seferinde tahmin eden bir model de yuksek accuracy
        # alabilir). Sinif basina precision/recall/f1 + kac ornek uzerinden
        # hesaplandigi (support), hangi kategorilerin gercekten ogrenilemedigini
        # gosteriyor.
        present = sorted(set(gold.tolist()) | set(preds.tolist()))
        precision, recall, f1, support = precision_recall_fscore_support(
            gold, preds, labels=present, average=None, zero_division=0
        )
        metrics = {"accuracy": float(acc), "macro_f1": float(f1.mean())}
        for idx, p, r, f, s in zip(present, precision, recall, f1, support):
            slug = _slugify_label(CLASSIFICATION_LABELS[idx])
            metrics[f"precision_{slug}"] = float(p)
            metrics[f"recall_{slug}"] = float(r)
            metrics[f"f1_{slug}"] = float(f)
            metrics[f"support_{slug}"] = int(s)
        return metrics

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


def evaluate_classification_report(
    texts: list[str], labels: list[int], model_dir: str | None = None
) -> str:
    """Verilen (texts, labels) uzerinde kayitli siniflandiriciyi calistirip
    sklearn'in standart sinif-basina precision/recall/f1/support tablosunu
    metin olarak dondurur -- README/CV'ye eklenebilecek, tek bir accuracy
    sayisindan cok daha durust bir ozet."""
    model_dir = model_dir or str(MODELS_DIR / "classifier")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()

    inputs = tokenizer(texts, truncation=True, padding=True, return_tensors="pt", max_length=256)
    with torch.no_grad():
        logits = model(**inputs).logits
    preds = torch.argmax(logits, dim=-1).tolist()

    present = sorted(set(labels) | set(preds))
    return _sk_classification_report(
        labels,
        preds,
        labels=present,
        target_names=[CLASSIFICATION_LABELS[i] for i in present],
        zero_division=0,
    )
