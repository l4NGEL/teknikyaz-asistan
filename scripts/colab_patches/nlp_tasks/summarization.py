"""Özetleme: abstractive (mT5 tabanlı, anlam üreten) + extractive (TextRank, hızlı/ucuz) modlar.

İki yaklaşımı birlikte sunmak bilinçli bir tasarım kararı: extractive özet GPU
gerektirmez ve halüsinasyon riski taşımaz (cümleler doğrudan kaynaktan alınır),
abstractive özet ise daha akıcıdır ama üretim (generation) riski taşır. Üretim
sisteminde ikisini karşılaştırmak, XAI/güvenilirlik tartışmasına doğal bir giriş sağlar.
"""
from __future__ import annotations

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from src.config import MODEL_CONFIG

_abstractive_model = None
_abstractive_tokenizer = None


def _get_abstractive_model():
    global _abstractive_model, _abstractive_tokenizer
    if _abstractive_model is None:
        model_name = MODEL_CONFIG.summarization_model
        _abstractive_tokenizer = AutoTokenizer.from_pretrained(model_name)
        _abstractive_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _abstractive_model.to(device)
    return _abstractive_model, _abstractive_tokenizer


def abstractive_summarize(text: str, max_length: int = 128, min_length: int = 32) -> str:
    model, tokenizer = _get_abstractive_model()
    device = next(model.parameters()).device
    inputs = tokenizer(
        text,
        max_length=784,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    generated_ids = model.generate(
        **inputs,
        num_beams=2,
        max_length=max_length,
        min_length=min_length,
        repetition_penalty=2.5,
        length_penalty=2.0,
        early_stopping=True,
    )
    return tokenizer.decode(generated_ids[0], skip_special_tokens=True)


def extractive_summarize(text: str, sentence_count: int = 3, language: str = "turkish") -> str:
    from sumy.nlp.tokenizers import Tokenizer
    from sumy.parsers.plaintext import PlaintextParser
    from sumy.summarizers.text_rank import TextRankSummarizer

    parser = PlaintextParser.from_string(text, Tokenizer(language))
    summarizer = TextRankSummarizer()
    sentences = summarizer(parser.document, sentence_count)
    return " ".join(str(s) for s in sentences)


def summarize(text: str, method: str = "extractive", **kwargs) -> str:
    if method == "extractive":
        return extractive_summarize(text, **kwargs)
    if method == "abstractive":
        return abstractive_summarize(text, **kwargs)
    raise ValueError(f"Bilinmeyen yöntem: {method} (extractive | abstractive)")
