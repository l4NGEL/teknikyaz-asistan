"""Niceleme (quantization): modeli daha az bit ile temsil ederek bellek/hız kazanır.

Üç seviye gösteriyoruz:
1. bitsandbytes 4-bit/8-bit — eğitim sırasında da kullanılabilir (QLoRA'da zaten gördük).
2. Dinamik INT8 (torch.quantization) — CPU çıkarım için hızlı ve bağımlılıksız bir seçenek.
3. GGUF dönüşümü (llama.cpp) — üretimde llama.cpp/Ollama gibi CPU-dostu runtime'larla
   servis etmek için; burada dönüşüm komutunu üreten bir yardımcı fonksiyon veriyoruz
   çünkü asıl dönüşüm ayrı bir derlenmiş araç (llama.cpp) gerektirir.
"""
from __future__ import annotations

from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def load_4bit(model_path: str):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    return AutoModelForCausalLM.from_pretrained(model_path, quantization_config=bnb_config, device_map="auto")


def load_8bit(model_path: str):
    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    return AutoModelForCausalLM.from_pretrained(model_path, quantization_config=bnb_config, device_map="auto")


def dynamic_int8_cpu(model_path: str):
    """CPU üzerinde çalıştırmak için PyTorch dinamik niceleme (bitsandbytes gerektirmez)."""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32)
    quantized = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    return quantized, tokenizer


def build_gguf_conversion_command(
    hf_model_dir: str, llama_cpp_dir: str, out_path: str, quant_type: str = "Q4_K_M"
) -> list[str]:
    """llama.cpp'nin convert + quantize adımlarını çalıştıracak komutu üretir (siz çalıştırırsınız).

    Colab'da llama.cpp'yi klonlayıp derledikten sonra bu komutu `subprocess.run(..., check=True)`
    ile çağırabilirsiniz. Burada komutu döndürmemizin nedeni: derleme adımı ortam bağımlı
    olduğu için otomatik çalıştırmak yerine kullanıcıya şeffaf bırakmak.
    """
    convert_script = str(Path(llama_cpp_dir) / "convert_hf_to_gguf.py")
    quantize_bin = str(Path(llama_cpp_dir) / "llama-quantize")
    fp16_out = str(Path(out_path).with_suffix(".fp16.gguf"))
    return [
        "python", convert_script, hf_model_dir, "--outfile", fp16_out, "--outtype", "f16",
        "&&", quantize_bin, fp16_out, out_path, quant_type,
    ]
