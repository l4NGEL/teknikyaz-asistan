"""Soru-cevap: extractive (bağlamdan span çıkarır) + generative (RAG üzerinden, bkz. src/rag).

Extractive QA'nın avantajı: cevap her zaman kaynak metinden birebir alınır, halüsinasyon
imkansızdır — ama sadece kaynakta *kelimesi kelimesine* geçen cevaplar için çalışır.
Generative QA (rag_pipeline.answer) daha esnektir ama XAI modülündeki groundedness
kontrolünden geçmelidir.
"""
from __future__ import annotations

import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

from src.config import MODEL_CONFIG

_qa_model = None
_qa_tokenizer = None


def _get_qa_model():
    global _qa_model, _qa_tokenizer
    if _qa_model is None:
        model_name = MODEL_CONFIG.qa_model
        _qa_tokenizer = AutoTokenizer.from_pretrained(model_name)
        _qa_model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _qa_model.to(device)
    return _qa_model, _qa_tokenizer


def extractive_answer(question: str, context: str) -> dict:
    model, tokenizer = _get_qa_model()
    device = next(model.parameters()).device

    inputs = tokenizer(
        question,
        context,
        return_tensors="pt",
        truncation="only_second",
        max_length=512,
        return_offsets_mapping=True,
    )
    offset_mapping = inputs.pop("offset_mapping")[0].tolist()
    sequence_ids = inputs.sequence_ids(0)

    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)

    start_logits = outputs.start_logits[0]
    end_logits = outputs.end_logits[0]
    input_ids = inputs["input_ids"][0]

    best_score = float("-inf")
    best_start = 0
    best_end = 0
    for start in range(len(start_logits)):
        for end in range(start, min(start + 64, len(end_logits))):
            if sequence_ids[start] != 1 or sequence_ids[end] != 1:
                continue
            score = start_logits[start].item() + end_logits[end].item()
            if score > best_score:
                best_score = score
                best_start = start
                best_end = end

    answer = tokenizer.decode(input_ids[best_start : best_end + 1], skip_special_tokens=True)

    start_char = None
    end_char = None
    for idx in range(best_start, best_end + 1):
        if sequence_ids[idx] != 1:
            continue
        char_start, char_end = offset_mapping[idx]
        if char_start == char_end:
            continue
        if start_char is None:
            start_char = char_start
        end_char = char_end

    if start_char is None:
        start_char = context.find(answer)
        end_char = start_char + len(answer) if start_char >= 0 else 0

    start_probs = torch.softmax(outputs.start_logits, dim=-1)
    end_probs = torch.softmax(outputs.end_logits, dim=-1)
    score = float(start_probs[0, best_start] * end_probs[0, best_end])

    return {
        "answer": answer,
        "score": score,
        "start": start_char,
        "end": end_char,
    }


if __name__ == "__main__":
    ctx = "kullanici_olustur fonksiyonu, e_posta parametresi zaten kayıtlıysa DuplicateEmailError fırlatır."
    print(extractive_answer("kullanici_olustur hangi hatayı fırlatır?", ctx))
