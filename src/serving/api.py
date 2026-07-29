"""FastAPI servis katmanı: RAG sohbet + tekil NLP görevleri, hepsi metriklenerek.

Her istek `metrics_store`'a yazılır (gecikme, halüsinasyon skoru) — bu, LLMOps dashboard'unun
(src/llmops/dashboard.py) beslendiği kaynaktır. Sohbet uç noktası ek olarak XAI
halüsinasyon kontrolünden (src/xai/hallucination_check.py) geçer.
"""
from __future__ import annotations

import time

from fastapi import FastAPI, HTTPException

from src.llmops.metrics_store import log_request
from src.serving.schemas import (
    ChatRequest,
    ChatResponse,
    ClassifyRequest,
    ClassifyResponse,
    NerRequest,
    NerResponse,
    SummarizeRequest,
    SummarizeResponse,
    TranslateRequest,
    TranslateResponse,
)

app = FastAPI(title="TİHA-Asistan API", version="0.1.0")

MODEL_VERSION = "base-rag-v1"  # fine-tune/DPO adaptörü devreye girince buradan güncellenir


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    from src.nlp_tasks.dialogue import chat as run_chat
    from src.xai.hallucination_check import groundedness_score

    start = time.perf_counter()
    try:
        result = run_chat(req.session_id, req.message)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    latency_ms = (time.perf_counter() - start) * 1000

    hallucination_score = None
    try:
        context_texts = result.get("context_texts", [])
        if context_texts:
            hallucination_score = 1.0 - groundedness_score(result["answer"], context_texts)
    except Exception:
        pass  # XAI kontrolü opsiyoneldir; başarısız olursa yanıtı engellemez

    log_request(
        endpoint="/chat", latency_ms=latency_ms, model_version=MODEL_VERSION,
        hallucination_score=hallucination_score,
        input_preview=req.message, output_preview=result["answer"],
    )

    return ChatResponse(
        session_id=result["session_id"],
        answer=result["answer"],
        sources=result["sources"],
        hallucination_score=hallucination_score,
        turn_count=result["turn_count"],
    )


@app.post("/classify", response_model=ClassifyResponse)
def classify(req: ClassifyRequest):
    from src.nlp_tasks.classification import zero_shot_classify

    start = time.perf_counter()
    result = zero_shot_classify(req.text)
    log_request(endpoint="/classify", latency_ms=(time.perf_counter() - start) * 1000, input_preview=req.text)
    return ClassifyResponse(**result)


@app.post("/ner", response_model=NerResponse)
def ner(req: NerRequest):
    from src.nlp_tasks.ner import extract_entities

    start = time.perf_counter()
    entities = extract_entities(req.text)
    log_request(endpoint="/ner", latency_ms=(time.perf_counter() - start) * 1000, input_preview=req.text)
    return NerResponse(entities=entities)


@app.post("/summarize", response_model=SummarizeResponse)
def summarize(req: SummarizeRequest):
    from src.nlp_tasks.summarization import summarize as run_summarize

    start = time.perf_counter()
    summary = run_summarize(req.text, method=req.method)
    log_request(endpoint="/summarize", latency_ms=(time.perf_counter() - start) * 1000, input_preview=req.text, output_preview=summary)
    return SummarizeResponse(summary=summary)


@app.post("/translate", response_model=TranslateResponse)
def translate(req: TranslateRequest):
    from src.nlp_tasks.translation import translate as run_translate

    start = time.perf_counter()
    translated = run_translate(req.text, direction=req.direction)
    log_request(endpoint="/translate", latency_ms=(time.perf_counter() - start) * 1000, input_preview=req.text, output_preview=translated)
    return TranslateResponse(translated_text=translated)
