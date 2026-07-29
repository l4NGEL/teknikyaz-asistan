"""Budama (pruning): ağırlık matrislerindeki en küçük (etkisiz) parametreleri sıfırlar.

Büyük bir causal LM'i budamak hem riskli hem de bu projenin kapsamı için ağırdır; bunun
yerine src/nlp_tasks/classification.py ile eğittiğiniz BERT sınıflandırıcı üzerinde,
budama-doğruluk ödünleşimini (trade-off) net ve hızlı biçimde gözlemleyebileceğiniz
çalışan bir örnek sunuyoruz. Aynı teknik (torch.nn.utils.prune) büyük modellere de
uygulanabilir; fark sadece ölçek ve süredir.
"""
from __future__ import annotations

import copy

import torch
from torch.nn.utils import prune
from transformers import AutoModelForSequenceClassification


def magnitude_prune(model: torch.nn.Module, amount: float = 0.3) -> torch.nn.Module:
    """Her Linear katmanında büyüklüğü (magnitude) en küçük ağırlıkların `amount` oranını sıfırlar."""
    pruned = copy.deepcopy(model)
    for module in pruned.modules():
        if isinstance(module, torch.nn.Linear):
            prune.l1_unstructured(module, name="weight", amount=amount)
            prune.remove(module, "weight")  # maskeyi kalıcı hale getir, hook'u kaldır
    return pruned


def sparsity_report(model: torch.nn.Module) -> dict:
    total, zeros = 0, 0
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            total += module.weight.numel()
            zeros += (module.weight == 0).sum().item()
    return {"total_params": total, "zero_params": zeros, "sparsity": zeros / total if total else 0.0}


if __name__ == "__main__":
    from src.config import MODELS_DIR

    model = AutoModelForSequenceClassification.from_pretrained(str(MODELS_DIR / "classifier"))
    pruned = magnitude_prune(model, amount=0.3)
    print("Budama öncesi:", sparsity_report(model))
    print("Budama sonrası:", sparsity_report(pruned))
