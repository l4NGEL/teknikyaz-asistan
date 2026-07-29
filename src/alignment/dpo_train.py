"""DPO (Direct Preference Optimization) ile hizalama.

RLHF'in klasik yolu (ödül modeli + PPO) iki ayrı model ve kararsız bir eğitim döngüsü
gerektirir. DPO, tercih çiftlerini (chosen/rejected) doğrudan bir kayıp fonksiyonuna
çevirerek aynı hizalama hedefine tek aşamada, çok daha kararlı şekilde ulaşır — bu yüzden
küçük ölçekli/Colab projelerinde PPO yerine tercih edilir. PPO'nun neden daha karmaşık
olduğunu (ayrı reward model + value model + rollout) burada deneyimlemeseniz de,
`docs`/notebook içinde karşılaştırmalı olarak anlatılır.

DPO, adım 4'te QLoRA ile eğitilmiş adaptörün ÜZERİNE devam eder (aynı LoRA katmanları).
"""
from __future__ import annotations

import json

from datasets import Dataset
from peft import PeftModel
from trl import DPOConfig, DPOTrainer

from src.config import MODELS_DIR, QLORA_CONFIG
from src.finetune.qlora_train import load_quantized_model_and_tokenizer


def load_preference_dataset(path=None) -> Dataset:
    from src.config import PROCESSED_DIR

    path = path or (PROCESSED_DIR / "dpo_pairs.jsonl")
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return Dataset.from_list(rows)


def train(
    sft_adapter_dir: str | None = None,
    output_dir: str | None = None,
    base_model: str | None = None,
) -> str:
    sft_adapter_dir = sft_adapter_dir or QLORA_CONFIG.output_dir
    output_dir = output_dir or str(MODELS_DIR / "dpo-adapter")

    base_model_obj, tokenizer = load_quantized_model_and_tokenizer(base_model)
    # QLoRA aşamasında eğitilen adaptörü yükleyip DPO ile devam ettiriyoruz.
    model = PeftModel.from_pretrained(base_model_obj, sft_adapter_dir, is_trainable=True)

    dataset = load_preference_dataset()

    dpo_config = DPOConfig(
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=5e-6,
        beta=0.1,  # KL-penalty katsayısı: referans modelden ne kadar uzaklaşabileceğini sınırlar
        logging_steps=10,
        bf16=True,
        report_to=["mlflow"],
    )

    trainer = DPOTrainer(
        model=model,
        args=dpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    return output_dir


if __name__ == "__main__":
    path = train()
    print(f"DPO adaptörü kaydedildi -> {path}")
