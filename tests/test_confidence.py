from src.xai.confidence import combine_confidence, confidence_bucket, lexical_groundedness, should_escalate


def test_lexical_groundedness_rewards_overlap():
    score = lexical_groundedness(
        "README kurulum ve lisans bölümlerini içerir.",
        ["README belgesi kurulum, kullanım ve lisans bölümlerinden oluşur."],
    )
    assert score > 0.3


def test_conflicts_always_escalate():
    escalate, reason = should_escalate(0.99, n_conflicts=1, threshold=0.45)
    assert escalate is True
    assert "çelişki" in reason


def test_low_score_escalates_with_bucket():
    escalate, reason = should_escalate(0.2, n_conflicts=0, threshold=0.45)
    assert escalate is True
    assert "düşük" in reason


def test_combine_confidence_is_bounded():
    contexts = [{"rerank_score": 0.8}]
    score = combine_confidence(0.9, contexts, n_conflicts=0)
    assert 0.0 <= score <= 1.0
    assert confidence_bucket(score) in {"yüksek", "orta", "düşük"}
