from src.rag.hierarchy import expand_hits, parent_id_of


def test_parent_id_prefers_metadata():
    assert parent_id_of({"chunk_id": "x::0", "metadata": {"parent_id": "README Şablonu"}}) == "README Şablonu"


def test_parent_id_falls_back_to_chunk_id():
    assert parent_id_of({"chunk_id": "README Şablonu::2", "text": "..."}) == "README Şablonu"


def test_expand_hits_joins_siblings_and_keeps_order():
    corpus = [
        {"chunk_id": "API::0", "doc_title": "API", "text": "bir", "metadata": {"chunk_index": 0, "parent_id": "API"}},
        {"chunk_id": "API::1", "doc_title": "API", "text": "iki", "metadata": {"chunk_index": 1, "parent_id": "API"}},
        {"chunk_id": "README::0", "doc_title": "README", "text": "üç", "metadata": {"chunk_index": 0, "parent_id": "README"}},
    ]
    hits = [corpus[1], corpus[2]]
    expanded = expand_hits(hits, corpus=corpus)

    assert [h["parent_id"] for h in expanded] == ["API", "README"]
    assert expanded[0]["parent_text"] == "bir\niki"
    assert expanded[0]["sibling_count"] == 2
    assert expanded[1]["parent_text"] == "üç"
