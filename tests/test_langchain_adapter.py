from unittest.mock import patch

from langchain_core.documents import Document

from src.rag.langchain_adapter import TeknikYazLLM, TeknikYazRetriever, build_qa_chain

_FAKE_CONTEXTS = [
    {
        "chunk_id": "c1",
        "text": "REST API dokümantasyonu bir OpenAPI şeması ile başlar.",
        "doc_title": "API Dokümantasyon Şablonu",
        "doc_url": "https://example.com/api-sablonu",
        "rerank_score": 0.91,
    },
    {
        "chunk_id": "c2",
        "text": "Her endpoint için istek/yanıt örnekleri verilmelidir.",
        "doc_title": "API Dokümantasyon Şablonu",
        "doc_url": "https://example.com/api-sablonu",
        "rerank_score": 0.77,
    },
]


def test_retriever_wraps_retrieve_output_as_documents():
    with patch("src.rag.langchain_adapter.retrieve", return_value=_FAKE_CONTEXTS) as mock_retrieve:
        retriever = TeknikYazRetriever(top_k=2, rerank=True)
        docs = retriever.invoke("REST API dokümantasyonu nasıl yazılır?")

    mock_retrieve.assert_called_once_with("REST API dokümantasyonu nasıl yazılır?", final_k=2, rerank=True)
    assert len(docs) == 2
    assert all(isinstance(d, Document) for d in docs)
    assert docs[0].page_content == _FAKE_CONTEXTS[0]["text"]
    assert docs[0].metadata["doc_title"] == "API Dokümantasyon Şablonu"


def test_retriever_returns_empty_list_when_no_contexts():
    with patch("src.rag.langchain_adapter.retrieve", return_value=[]):
        docs = TeknikYazRetriever().invoke("bağlamda karşılığı olmayan bir soru")

    assert docs == []


def test_llm_call_builds_chat_messages_and_delegates_to_generate():
    with patch("src.rag.langchain_adapter.generate", return_value="Üretilen yanıt.") as mock_generate:
        llm = TeknikYazLLM(model_path="checkpoints/tiha-asistan-dpo", max_new_tokens=128)
        result = llm.invoke("Bağlam:\n...\n\nSoru: Nasıl?")

    assert result == "Üretilen yanıt."
    messages = mock_generate.call_args.args[0]
    assert messages[0]["role"] == "system"
    assert messages[1] == {"role": "user", "content": "Bağlam:\n...\n\nSoru: Nasıl?"}
    assert mock_generate.call_args.kwargs["model_path"] == "checkpoints/tiha-asistan-dpo"
    assert mock_generate.call_args.kwargs["max_new_tokens"] == 128


def test_build_qa_chain_returns_answer_and_sources():
    with (
        patch("src.rag.langchain_adapter.retrieve", return_value=_FAKE_CONTEXTS),
        patch(
            "src.rag.langchain_adapter.generate",
            return_value="Cevap burada. [Kaynak: API Dokümantasyon Şablonu]",
        ),
    ):
        chain = build_qa_chain(top_k=2)
        result = chain.invoke("REST API dokümantasyonu nasıl yazılır?")

    assert result["answer"] == "Cevap burada. [Kaynak: API Dokümantasyon Şablonu]"
    assert result["sources"] == [
        {"title": "API Dokümantasyon Şablonu", "url": "https://example.com/api-sablonu"},
        {"title": "API Dokümantasyon Şablonu", "url": "https://example.com/api-sablonu"},
    ]
