import os

import pytest

from src.rag import es_index


class _FakeClient:
    def __init__(self, hits):
        self._hits = hits
        self.last_body = None

    def search(self, index, **body):
        self.last_body = body
        return {"hits": {"hits": self._hits}}


def _doc(chunk_id, title, score):
    return {"_score": score, "_source": {"chunk_id": chunk_id, "doc_title": title, "doc_url": None, "text": "gövde"}}


def test_mapping_has_vector_and_three_text_fields():
    props = es_index.index_mapping()["mappings"]["properties"]
    assert props["embedding"] == {"type": "dense_vector", "dims": 1024, "index": True, "similarity": "cosine"}
    assert props["title_text"]["analyzer"] == "turkish"
    assert props["title_text_std"]["analyzer"] == "standard"
    assert props["title_text_f5"]["analyzer"] == "turkish_f5"


def test_f5_analyzer_lowercases_turkish_and_truncates_to_five():
    analysis = es_index.index_mapping()["settings"]["analysis"]
    assert analysis["analyzer"]["turkish_f5"]["filter"] == ["apostrophe", "tr_lowercase", "tr_stop", "prefix5"]
    assert analysis["filter"]["tr_lowercase"]["language"] == "turkish"
    assert analysis["filter"]["prefix5"]["length"] == 5


def test_actions_index_title_with_body_in_every_text_field():
    chunk = {"chunk_id": "README Şablonu::0", "doc_title": "README Şablonu", "text": "## Kurulum", "doc_url": None}
    (action,) = list(es_index.build_actions([chunk], [[0.1, 0.2]], index="idx"))
    src = action["_source"]
    assert action["_id"] == "README Şablonu::0"
    assert src["title_text"] == src["title_text_std"] == src["title_text_f5"] == "README Şablonu ## Kurulum"
    assert src["embedding"] == [0.1, 0.2]


def test_bm25_hits_match_local_bm25_shape(monkeypatch):
    fake = _FakeClient([_doc("a", "README Şablonu", 7.5)])
    monkeypatch.setattr(es_index, "get_client", lambda: fake)
    hits = es_index.query_bm25("readme", top_k=3)
    assert hits == [{"chunk_id": "a", "text": "gövde", "doc_title": "README Şablonu", "doc_url": None, "bm25_score": 7.5}]
    assert fake.last_body["query"] == {"match": {"title_text_f5": {"query": "readme"}}}
    assert fake.last_body["size"] == 3


def test_dense_score_converted_back_to_cosine(monkeypatch):
    # ES cosine benzerliği (1 + cos) / 2 olarak döner; 0.9 -> cos 0.8
    fake = _FakeClient([_doc("a", "API Dokümantasyonu Şablonu", 0.9)])
    monkeypatch.setattr(es_index, "get_client", lambda: fake)
    monkeypatch.setattr(es_index, "embed_query", lambda text: [0.0, 1.0])
    (hit,) = es_index.query_dense("api", top_k=5)
    assert hit["similarity"] == pytest.approx(0.8)
    assert "es_score" not in hit
    assert fake.last_body["knn"]["k"] == 5
    assert fake.last_body["knn"]["num_candidates"] >= 100


def _live_client():
    if not os.environ.get("ES_URL"):
        pytest.skip("ES_URL tanımlı değil; canlı Elasticsearch testi atlandı")
    client = es_index.get_client()
    if not client.ping():
        pytest.skip("Elasticsearch erişilemiyor")
    return client


def test_live_snowball_splits_word_forms_that_f5_keeps_together():
    """Ölçümde görülen sorun: Snowball 'şablonu' ve 'şablonunda'yı farklı köke indiriyor."""
    client = _live_client()
    index = "teknikyaz_analyzer_test"
    if client.indices.exists(index=index):
        client.indices.delete(index=index)
    client.indices.create(index=index, **es_index.index_mapping(dims=2))
    try:
        def tokens(field, text):
            return [t["token"] for t in client.indices.analyze(index=index, field=field, text=text)["tokens"]]

        assert tokens("title_text", "Şablonu") != tokens("title_text", "şablonunda")
        assert tokens("title_text_f5", "Şablonu") == tokens("title_text_f5", "şablonunda") == ["şablo"]
        assert tokens("title_text_f5", "README'de") == tokens("title_text_f5", "readme")
    finally:
        client.indices.delete(index=index)
