"""FastAPI istek/yanıt şemaları (Pydantic)."""
from __future__ import annotations

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    session_id: str = Field(..., description="Diyalog oturumu kimliği")
    message: str = Field(..., min_length=1, max_length=2000)


class SourceRef(BaseModel):
    title: str
    url: str


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    sources: list[SourceRef]
    hallucination_score: float | None = None
    turn_count: int


class ClassifyRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)


class ClassifyResponse(BaseModel):
    label: str
    scores: dict[str, float]


class NerRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)


class NerEntity(BaseModel):
    entity_group: str
    word: str
    start: int
    end: int
    score: float
    source: str


class NerResponse(BaseModel):
    entities: list[NerEntity]


class SummarizeRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000)
    method: str = Field(default="extractive", pattern="^(extractive|abstractive)$")


class SummarizeResponse(BaseModel):
    summary: str


class TranslateRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)
    direction: str = Field(default="tr-en", pattern="^(tr-en|en-tr)$")


class TranslateResponse(BaseModel):
    translated_text: str
