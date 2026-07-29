"""QLoRA (4-bit quantized LoRA) ile instruction fine-tuning.

Neden QLoRA: Colab'ın ücretsiz T4 GPU'sunda (~15GB VRAM) birkaç milyar parametreli bir
modeli tam hassasiyetle (full fine-tuning) eğitmek mümkün değildir. QLoRA, temel modeli
4-bit'e sıkıştırıp (NF4) sadece küçük LoRA adaptör katmanlarını eğiterek bunu mümkün kılar
— parametrelerin <%1'ini güncelleyerek tam fine-tuning'e yakın performans elde edilir.
"""
from __future__ import annotations

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


def train(base_model: str | None = None, output_dir: str | None = None) -> str:
    output_dir = output_dir or QLORA_CONFIG.output_dir

    model, tokenizer = load_quantized_model_and_tokenizer(base_model)
    peft_model = build_lora_model(model)

    train_ds, eval_ds = build_hf_dataset()

    sft_config = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=QLORA_CONFIG.num_train_epochs,
        per_device_train_batch_size=QLORA_CONFIG.per_device_train_batch_size,
        gradient_accumulation_steps=QLORA_CONFIG.gradient_accumulation_steps,
        learning_rate=QLORA_CONFIG.learning_rate,
        max_seq_length=QLORA_CONFIG.max_seq_length,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        bf16=True,
        report_to=["mlflow"],  # bkz. src/llmops/experiment_tracker.py
    )

    trainer = SFTTrainer(
        model=peft_model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    return output_dir


if __name__ == "__main__":
    path = train()
    print(f"LoRA adaptörü kaydedildi -> {path}")
