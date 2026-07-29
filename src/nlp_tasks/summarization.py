"""Özetleme: abstractive (mT5 tabanlı, anlam üreten) + extractive (TextRank, hızlı/ucuz) modlar.

İki yaklaşımı birlikte sunmak bilinçli bir tasarım kararı: extractive özet GPU
gerektirmez ve halüsinasyon riski taşımaz (cümleler doğrudan kaynaktan alınır),
abstractive özet ise daha akıcıdır ama üretim (generation) riski taşır. Üretim
sisteminde ikisini karşılaştırmak, XAI/güvenilirlik tartışmasına doğal bir giriş sağlar.
"""
from __future__ import annotations

from transformers import pipeline

from src.config import MODEL_CONFIG

_abstractive_pipe = None


def _get_abstractive_pipe():
    global _abstractive_pipe
    if _abstractive_pipe is None:
        _abstractive_pipe = pipeline("summarization", model=MODEL_CONFIG.summarization_model)
    return _abstractive_pipe


def abstractive_summarize(text: str, max_length: int = 128, min_length: int = 32) -> str:
    pipe = _get_abstractive_pipe()
    out = pipe(text, max_length=max_length, min_length=min_length, do_sample=False)
    return out[0]["summary_text"]


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
