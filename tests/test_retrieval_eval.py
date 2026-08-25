from src.rag.eval import evaluate_hits, evaluate_retriever, precision_at_k, recall_at_k


def test_precision_and_recall_at_k():
    retrieved = ["README Şablonu", "SSS Şablonu", "API Dokümantasyonu Şablonu"]
    relevant = {"README Şablonu"}

    assert precision_at_k(retrieved, relevant, 1) == 1.0
    assert precision_at_k(retrieved, relevant, 3) == 1 / 3
    assert recall_at_k(retrieved, relevant, 1) == 1.0
    assert recall_at_k(retrieved, relevant, 3) == 1.0


def test_recall_is_zero_when_relevant_missing():
    assert recall_at_k(["API"], {"README Şablonu"}, 5) == 0.0


def test_evaluate_retriever_aggregates_mean_metrics():
    gold = [
        {"query": "readme nedir", "relevant_titles": ["README Şablonu"]},
        {"query": "api nasıl yazılır", "relevant_titles": ["API Dokümantasyonu Şablonu"]},
    ]

    def fake_retrieve(query: str, final_k: int = 5, **_kwargs):
        if "readme" in query:
            titles = ["README Şablonu", "SSS Şablonu"]
        else:
            titles = ["Kurulum Kılavuzu Şablonu", "API Dokümantasyonu Şablonu"]
        return [{"doc_title": t, "text": t} for t in titles[:final_k]]

    report = evaluate_retriever(fake_retrieve, gold=gold, k_values=(1, 2))

    assert report.n_queries == 2
    assert report.mean_precision[1] == 0.5  # README hit, API miss at rank 1
    assert report.mean_recall[2] == 1.0
    assert "P@k" in report.summary_table()


def test_evaluate_hits_uses_doc_title():
    hits = [{"doc_title": "README Şablonu"}, {"doc_title": "API Dokümantasyonu Şablonu"}]
    result = evaluate_hits(hits, ["README Şablonu"], k=2)
    assert result.precision == 0.5
    assert result.recall == 1.0
    assert result.hit is True
