from src.rag.fusion import (
    fuse_candidates,
    merge_hit_dicts,
    rrf_scores,
    template_boost,
    title_overlap,
    wants_template,
)


def test_merge_keeps_bm25_score_on_dense_hit():
    dense = [{"chunk_id": "a", "text": "t", "doc_title": "README Şablonu", "similarity": 0.4}]
    bm25 = [{"chunk_id": "a", "text": "t", "doc_title": "README Şablonu", "bm25_score": 9.1}]
    merged = merge_hit_dicts(dense, bm25)
    assert merged["a"]["similarity"] == 0.4
    assert merged["a"]["bm25_score"] == 9.1


def test_rrf_prefers_higher_ranks():
    scores = rrf_scores(["first", "second", "third"])
    assert scores["first"] > scores["second"] > scores["third"]


def test_sss_how_to_write_query_is_template_intent():
    assert wants_template("SSS belgesi hangi soruları kapsamalı?")
    question = "Bir README'de hangi bölümler olmalı?"
    assert wants_template(question)
    assert template_boost(question, "README Şablonu") == 1.0
    assert template_boost(question, "README — Bildirim Servisi") == 0.0
    assert 0 < template_boost(question, "Kod Stil Rehberi Şablonu") < 1.0


def test_title_overlap_uses_readme_token():
    assert title_overlap("Bir README'de hangi bölümler olmalı?", "README Şablonu") > 0


def test_fusion_ranks_template_above_unrelated_when_query_asks_how_to_write():
    question = "Bir README'de hangi bölümler olmalı?"
    dense = [
        {"chunk_id": "style", "doc_title": "Kod Stil Rehberi Şablonu", "text": "stil"},
        {"chunk_id": "readme", "doc_title": "README Şablonu", "text": "kurulum kullanım lisans"},
    ]
    bm25 = [
        {"chunk_id": "readme", "doc_title": "README Şablonu", "text": "kurulum", "bm25_score": 12.0},
        {"chunk_id": "style", "doc_title": "Kod Stil Rehberi Şablonu", "text": "stil", "bm25_score": 1.0},
    ]
    ranked = fuse_candidates(question, dense, bm25, rerank_scores={"style": 0.99, "readme": 0.10})
    assert ranked[0]["doc_title"] == "README Şablonu"
