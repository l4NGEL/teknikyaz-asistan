"""retrieve() ACL entegrasyonu: yetkisiz aday füzyona ve reranker'a hiç ulaşmamalı."""
import pytest

from src.governance.acl import load_principals, principal_scope
from src.llmops import tracing
from src.rag import retriever

CORPUS = [
    ("Güvenlik Politikası Şablonu", "parola politikası ve erişim kontrolü"),
    ("API Referansı: oturum_dogrula", "oturum doğrulama uç noktası parola"),
    ("README Şablonu", "kurulum ve kullanım parola"),
    ("SSS — Hesap Güvenliği", "parolamı nasıl sıfırlarım"),
]


@pytest.fixture
def fake_index(monkeypatch, tmp_path):
    monkeypatch.setattr(tracing, "TRACES_PATH", tmp_path / "traces.jsonl")

    def hits(top_k, score_key):
        return [
            {"chunk_id": f"{title}::0", "doc_title": title, "doc_url": "", "text": text, score_key: 1.0 - i / 10}
            for i, (title, text) in enumerate(CORPUS[:top_k])
        ]

    monkeypatch.setattr(retriever.vector_store, "query", lambda q, top_k: hits(top_k, "similarity"))
    monkeypatch.setattr(retriever.bm25_index, "query", lambda q, top_k: hits(top_k, "bm25_score"))

    seen_by_reranker = []

    class FakeReranker:
        def predict(self, pairs):
            seen_by_reranker.extend(text for _, text in pairs)
            return [0.5] * len(pairs)

    monkeypatch.setattr(retriever, "_get_reranker", lambda: FakeReranker())
    return seen_by_reranker


def test_no_principal_keeps_legacy_behaviour(fake_index):
    hits, denied = retriever.retrieve_with_report("parola", final_k=10, hierarchical=False)
    assert len(hits) == 4 and denied == 0


def test_restricted_docs_never_reach_reranker(fake_index):
    musteri = load_principals()["musteri"]
    hits, denied = retriever.retrieve_with_report("parola", final_k=10, hierarchical=False, principal=musteri)
    titles = {h["doc_title"] for h in hits}
    assert titles == {"README Şablonu", "SSS — Hesap Güvenliği"}
    assert denied == 4  # 2 belge x (dense + BM25)
    assert "parola politikası ve erişim kontrolü" not in fake_index
    assert "oturum doğrulama uç noktası parola" not in fake_index


def test_request_scoped_principal_is_used(fake_index):
    with principal_scope(load_principals()["gelistirici"]):
        hits = retriever.retrieve("parola", final_k=10, hierarchical=False)
    assert "API Referansı: oturum_dogrula" in {h["doc_title"] for h in hits}
    assert "Güvenlik Politikası Şablonu" not in {h["doc_title"] for h in hits}


def test_retrieve_emits_stage_spans(fake_index, tmp_path):
    retriever.retrieve("parola", final_k=2, hierarchical=False, principal=load_principals()["musteri"])
    [trace] = tracing.load_traces(tmp_path / "traces.jsonl")
    assert [c["name"] for c in trace["children"]] == ["dense", "bm25", "acl_filter", "rerank", "fusion"]
    assert trace["attributes"]["denied_candidates"] == 4
