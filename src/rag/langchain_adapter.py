"""LangChain uyumluluk katmanı.

Çekirdek RAG pipeline'ı (`retriever.py`, `rag_pipeline.py`) bilinçli olarak framework-free
yazıldı: hybrid retrieve-then-rerank ve yerel model serving mantığının LangChain'in
soyutlamalarının arkasına gizlenmeden nasıl çalıştığını göstermek için. Bu modül o mantığı
tekrar yazmadan LangChain'in `Runnable` arayüzlerine (`BaseRetriever`, `LLM`) bağlıyor —
tek bağımlılık `langchain-core` (tüm `langchain` meta-paketi değil), çünkü ihtiyaç duyulan
tek şey Runnable/Document soyutlamaları.
"""
from __future__ import annotations

from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun, CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.language_models.llms import LLM
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough

from src.rag.rag_pipeline import SYSTEM_PROMPT, generate
from src.rag.retriever import retrieve


class TeknikYazRetriever(BaseRetriever):
    """`retriever.retrieve` (Chroma + cross-encoder reranking) için ince LangChain sarmalayıcısı."""

    top_k: int = 5
    rerank: bool = True

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        contexts = retrieve(query, final_k=self.top_k, rerank=self.rerank)
        return [
            Document(
                page_content=c["text"],
                metadata={
                    "chunk_id": c.get("chunk_id"),
                    "doc_title": c.get("doc_title"),
                    "doc_url": c.get("doc_url"),
                    "rerank_score": c.get("rerank_score"),
                },
            )
            for c in contexts
        ]


class TeknikYazLLM(LLM):
    """`rag_pipeline.generate` (yerel QLoRA+DPO modeli) için ince LangChain sarmalayıcısı.

    Yalnızca `_call`'ı implemente eder; asıl model yükleme/inference mantığı `rag_pipeline`
    içindeki `lru_cache`'li `_load_model`'de kalır, burada tekrarlanmaz.
    """

    model_path: str | None = None
    max_new_tokens: int = 300

    @property
    def _llm_type(self) -> str:
        return "teknikyaz-local"

    def _call(
        self,
        prompt: str,
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        return generate(messages, model_path=self.model_path, max_new_tokens=self.max_new_tokens)


def _format_docs(docs: list[Document]) -> str:
    return "\n\n".join(f"[{d.metadata.get('doc_title')}]\n{d.page_content}" for d in docs)


def _to_prompt(x: dict) -> str:
    return f"Bağlam:\n{_format_docs(x['documents'])}\n\nSoru: {x['question']}"


def build_qa_chain(model_path: str | None = None, top_k: int = 5, rerank: bool = True):
    """`rag_pipeline.answer()` ile aynı sonucu (answer + sources) üreten, LCEL ile kurulmuş
    uçtan uca zincir. Retrieve/rerank/generate mantığı değişmedi; sadece LangChain'in
    `Runnable` kompozisyonuna (`RunnableParallel`/`RunnableLambda`/`RunnablePassthrough`)
    bağlandı — `chain.invoke("soru")` -> {"answer": str, "sources": [{"title", "url"}, ...]}.
    """
    retriever = TeknikYazRetriever(top_k=top_k, rerank=rerank)
    llm = TeknikYazLLM(model_path=model_path)

    fetch = RunnableParallel(documents=retriever, question=RunnablePassthrough())
    return fetch | RunnableParallel(
        answer=RunnableLambda(_to_prompt) | llm,
        sources=RunnableLambda(
            lambda x: [
                {"title": d.metadata.get("doc_title"), "url": d.metadata.get("doc_url")}
                for d in x["documents"]
            ]
        ),
    )
