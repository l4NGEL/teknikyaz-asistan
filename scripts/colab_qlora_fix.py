#!/usr/bin/env python3
"""Eski zip icin tek dosyalik qlora_train yamasi (colab_patches gerektirmez)."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DST = ROOT / "src/finetune/qlora_train.py"

CONTENT = '''"""QLoRA (4-bit quantized LoRA) ile instruction fine-tuning."""
from __future__ import annotations

import inspect

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

from src.config import MODEL_CONFIG, QLORA_CONFIG
from src.finetune.dataset_utils import build_hf_dataset


def load_quantized_model_and_tokenizer(base_model: str | None = None):
    base_model = base_model or MODEL_CONFIG.base_llm

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model = prepare_model_for_kbit_training(model)
    return model, tokenizer


def build_lora_model(model):
    lora_config = LoraConfig(
        r=QLORA_CONFIG.lora_r,
        lora_alpha=QLORA_CONFIG.lora_alpha,
        lora_dropout=QLORA_CONFIG.lora_dropout,
        target_modules=list(QLORA_CONFIG.target_modules),
        bias="none",
        task_type="CAUSAL_LM",
    )
    peft_model = get_peft_model(model, lora_config)
    peft_model.print_trainable_parameters()
    return peft_model


def _seq_length_config_kwargs() -> dict:
    max_len = QLORA_CONFIG.max_seq_length
    params = inspect.signature(SFTConfig.__init__).parameters
    if "max_length" in params:
        return {"max_length": max_len}
    if "max_seq_length" in params:
        return {"max_seq_length": max_len}
    return {}


def _seq_length_trainer_kwargs() -> dict:
    if _seq_length_config_kwargs():
        return {}
    max_len = QLORA_CONFIG.max_seq_length
    params = inspect.signature(SFTTrainer.__init__).parameters
    if "max_seq_length" in params:
        return {"max_seq_length": max_len}
    if "max_length" in params:
        return {"max_length": max_len}
    return {}


def train(base_model: str | None = None, output_dir: str | None = None) -> str:
    output_dir = output_dir or QLORA_CONFIG.output_dir

    model, tokenizer = load_quantized_model_and_tokenizer(base_model)
    peft_model = build_lora_model(model)

    train_ds, eval_ds = build_hf_dataset()

    sft_kwargs = {
        "output_dir": output_dir,
        "num_train_epochs": QLORA_CONFIG.num_train_epochs,
        "per_device_train_batch_size": QLORA_CONFIG.per_device_train_batch_size,
        "gradient_accumulation_steps": QLORA_CONFIG.gradient_accumulation_steps,
        "learning_rate": QLORA_CONFIG.learning_rate,
        "logging_steps": 10,
        "eval_strategy": "epoch",
        "save_strategy": "epoch",
        "bf16": True,
        "report_to": ["mlflow"],
    }
    sft_kwargs.update(_seq_length_config_kwargs())

    config_params = inspect.signature(SFTConfig.__init__).parameters
    if "assistant_only_loss" in config_params:
        sft_kwargs["assistant_only_loss"] = True

    sft_config = SFTConfig(**sft_kwargs)

    trainer_params = inspect.signature(SFTTrainer.__init__).parameters
    trainer_kwargs = {
        "model": peft_model,
        "args": sft_config,
        "train_dataset": train_ds,
        "eval_dataset": eval_ds,
    }
    trainer_kwargs.update(_seq_length_trainer_kwargs())
    if "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = SFTTrainer(**trainer_kwargs)
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    return output_dir


if __name__ == "__main__":
    path = train()
    print(f"LoRA adaptoru kaydedildi -> {path}")
'''


def main() -> None:
    DST.parent.mkdir(parents=True, exist_ok=True)
    DST.write_text(CONTENT, encoding="utf-8")
    print(f"Guncellendi: {DST.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
