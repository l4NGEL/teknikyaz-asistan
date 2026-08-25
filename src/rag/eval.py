"""Retrieval değerlendirmesi: Precision@k ve Recall@k.

Gold etiketler belge başlığı düzeyindedir (bir chunk, `doc_title` gold kümedeyse
alakalı sayılır). Bu, Diagnis ilanındaki "Precision@k / Recall@k ile ölçümleme"
maddesinin çalışır karşılığıdır — küçük bir portföy korpusunda bile retrieval
değişikliklerinin nicel etkisini görmek için.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from src.config import RAG_CONFIG


@dataclass
class QueryResult:
    query: str
    k: int
    precision: float
    recall: float
    hit: bool
    retrieved_titles: list[str]
    relevant_titles: list[str]


@dataclass
class EvalReport:
    n_queries: int
    k_values: list[int]
    mean_precision: dict[int, float]
    mean_recall: dict[int, float]
    mean_hit_rate: dict[int, float]
    per_query: list[QueryResult]

    def to_dict(self) -> dict:
        payload = asdict(self)
        return payload

    def summary_table(self) -> str:
        lines = [f"{'k':<6}{'P@k':<12}{'R@k':<12}{'Hit@k':<12}"]
        for k in self.k_values:
            lines.append(
                f"{k:<6}{self.mean_precision[k]:<12.3f}"
                f"{self.mean_recall[k]:<12.3f}{self.mean_hit_rate[k]:<12.3f}"
            )
        return "\n".join(lines)


def load_gold(path: str | Path | None = None) -> list[dict]:
    path = Path(path or RAG_CONFIG.eval_gold_path)
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def precision_at_k(retrieved_titles: list[str], relevant: set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    top = retrieved_titles[:k]
    return sum(1 for title in top if title in relevant) / k


def recall_at_k(retrieved_titles: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    hit_titles = {title for title in retrieved_titles[:k] if title in relevant}
    return len(hit_titles) / len(relevant)


def hit_at_k(retrieved_titles: list[str], relevant: set[str], k: int) -> bool:
    return any(title in relevant for title in retrieved_titles[:k])


def evaluate_hits(hits: list[dict], relevant_titles: list[str], k: int) -> QueryResult:
    retrieved = [h.get("doc_title") or "" for h in hits]
    relevant = set(relevant_titles)
    return QueryResult(
        query="",
        k=k,
        precision=precision_at_k(retrieved, relevant, k),
        recall=recall_at_k(retrieved, relevant, k),
        hit=hit_at_k(retrieved, relevant, k),
        retrieved_titles=retrieved[:k],
        relevant_titles=list(relevant_titles),
    )


def evaluate_retriever(
    retrieve_fn,
    gold: list[dict] | None = None,
    k_values: tuple[int, ...] | None = None,
) -> EvalReport:
    """retrieve_fn(query, final_k) -> list[dict] (retriever.retrieve ile aynı şekil)."""
    gold = gold if gold is not None else load_gold()
    k_values = tuple(k_values or RAG_CONFIG.eval_k_values)
    per_query: list[QueryResult] = []

    for row in gold:
        query = row["query"]
        relevant = list(row["relevant_titles"])
        max_k = max(k_values)
        hits = retrieve_fn(query, final_k=max_k)
        for k in k_values:
            result = evaluate_hits(hits, relevant, k)
            result.query = query
            per_query.append(result)

    mean_p: dict[int, float] = {}
    mean_r: dict[int, float] = {}
    mean_h: dict[int, float] = {}
    for k in k_values:
        subset = [q for q in per_query if q.k == k]
        n = max(len(subset), 1)
        mean_p[k] = sum(q.precision for q in subset) / n
        mean_r[k] = sum(q.recall for q in subset) / n
        mean_h[k] = sum(1.0 if q.hit else 0.0 for q in subset) / n

    return EvalReport(
        n_queries=len(gold),
        k_values=list(k_values),
        mean_precision=mean_p,
        mean_recall=mean_r,
        mean_hit_rate=mean_h,
        per_query=per_query,
    )
