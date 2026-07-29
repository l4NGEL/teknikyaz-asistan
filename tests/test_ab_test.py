from src.llmops.ab_test import paired_ttest


def test_identical_scores_are_not_significant():
    scores = [0.8, 0.75, 0.9, 0.6, 0.85]
    result = paired_ttest(scores, list(scores))
    assert result.significant is False
    assert result.winner is None


def test_clearly_better_variant_b_is_detected():
    scores_a = [0.5, 0.52, 0.48, 0.51, 0.49, 0.50, 0.53, 0.47]
    scores_b = [0.9, 0.92, 0.88, 0.91, 0.89, 0.90, 0.93, 0.87]

    result = paired_ttest(scores_a, scores_b)

    assert result.significant is True
    assert result.winner == "B"


def test_mismatched_lengths_raise():
    import pytest

    with pytest.raises(ValueError):
        paired_ttest([0.1, 0.2], [0.1])
