"""İki model varyantını aynı değerlendirme seti üzerinde karşılaştırır (A/B test) ve
istatistiksel anlamlılığı basit bir t-test ile kontrol eder.

Neden istatistiksel test: iki modelin ortalama skorları farklı diye "daha iyi" demek
yanıltıcı olabilir — fark, örneklem gürültüsünden kaynaklanıyor olabilir. Bağımlı
örneklem t-testi (paired t-test) aynı soru setinde iki modelin skor farklarını
karşılaştırdığı için buraya uygundur.
"""
from __future__ import annotations

from dataclasses import dataclass

from scipy import stats  # scikit-learn bağımlılığıyla birlikte gelir

from src.llmops.experiment_tracker import log_metrics, run


@dataclass
class ABTestResult:
    variant_a_name: str
    variant_b_name: str
    mean_a: float
    mean_b: float
    p_value: float
    significant: bool
    winner: str | None


def paired_ttest(scores_a: list[float], scores_b: list[float], alpha: float = 0.05) -> ABTestResult:
    if len(scores_a) != len(scores_b):
        raise ValueError("İki skor listesi aynı uzunlukta olmalı (eşleştirilmiş test)")

    _t_stat, p_value = stats.ttest_rel(scores_b, scores_a)
    mean_a = sum(scores_a) / len(scores_a)
    mean_b = sum(scores_b) / len(scores_b)
    significant = bool(p_value < alpha)
    winner = None
    if significant:
        winner = "B" if mean_b > mean_a else "A"

    return ABTestResult(
        variant_a_name="A", variant_b_name="B",
        mean_a=mean_a, mean_b=mean_b, p_value=float(p_value),
        significant=significant, winner=winner,
    )


def run_ab_test(
    variant_a_name: str,
    variant_b_name: str,
    eval_fn_a,
    eval_fn_b,
    eval_examples: list,
    score_fn,
) -> ABTestResult:
    """eval_fn_*(example) -> model çıktısı; score_fn(example, output) -> float skor."""
    scores_a = [score_fn(ex, eval_fn_a(ex)) for ex in eval_examples]
    scores_b = [score_fn(ex, eval_fn_b(ex)) for ex in eval_examples]

    result = paired_ttest(scores_a, scores_b)

    with run(f"ab-test-{variant_a_name}-vs-{variant_b_name}"):
        log_metrics({
            f"{variant_a_name}_mean_score": result.mean_a,
            f"{variant_b_name}_mean_score": result.mean_b,
            "p_value": result.p_value,
        })

    return result
