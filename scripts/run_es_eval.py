"""Elasticsearch arka ucunu mevcut yerel arka uçla (Chroma + rank_bm25) aynı gold
sette karşılaştırır: tek tek bileşenler (BM25, dense) ve tam hibrit hat.

Gerekenler: çalışan bir Elasticsearch (ör. docker, xpack.security.enabled=false),
data/processed/chunks.jsonl ve dolu Chroma indeksi (bkz. run_retrieval_eval.py).
    python scripts/run_es_eval.py            # indeksi yeniden kurar ve ölçer
    python scripts/run_es_eval.py --no-index # mevcut ES indeksini kullanır
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.rag import bm25_index, es_index, vector_store  # noqa: E402
from src.rag.eval import evaluate_retriever, load_gold  # noqa: E402
from src.rag.retriever import retrieve  # noqa: E402

OUT = ROOT / "data" / "eval" / "es_comparison.json"


def mrr_at_10(report) -> float:
    ranks = []
    for q in report.per_query:
        if q.k != 10:
            continue
        rel = set(q.relevant_titles)
        ranks.append(next((1.0 / (i + 1) for i, t in enumerate(q.retrieved_titles) if t in rel), 0.0))
    return sum(ranks) / max(len(ranks), 1)


def timed(fn):
    """retrieve_fn'i sarar, her çağrının süresini (ms) kaydeder."""
    latencies: list[float] = []

    def wrapper(query: str, final_k: int = 10, **_kwargs):
        start = time.perf_counter()
        hits = fn(query, final_k)
        latencies.append((time.perf_counter() - start) * 1000)
        return hits

    return wrapper, latencies


CONFIGS = {
    "bm25_rank_bm25": lambda q, k: bm25_index.query(q, top_k=k),
    "bm25_es_standard": lambda q, k: es_index.query_bm25(q, top_k=k, field="title_text_std"),
    "bm25_es_turkish": lambda q, k: es_index.query_bm25(q, top_k=k, field="title_text"),
    "bm25_es_f5": lambda q, k: es_index.query_bm25(q, top_k=k, field="title_text_f5"),
    "dense_chroma": lambda q, k: vector_store.query(q, top_k=k),
    "dense_es_knn": lambda q, k: es_index.query_dense(q, top_k=k),
    "hybrid_local": lambda q, k: retrieve(q, final_k=k, hierarchical=False, backend="local"),
    "hybrid_es": lambda q, k: retrieve(q, final_k=k, hierarchical=False, backend="elasticsearch"),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-index", action="store_true")
    args = parser.parse_args()

    if not args.no_index:
        n = es_index.index_chunks()
        print(f"ES indeksine {n} chunk yazıldı ({es_index.INDEX_NAME})")

    gold = load_gold()
    # Embedding ve reranker modellerinin ilk yüklenmesi gecikmeyi bozmasın diye ısındır.
    for fn in CONFIGS.values():
        fn(gold[0]["query"], 10)

    results = {}
    print(f"{'yapılandırma':<20}{'Hit@1':>8}{'Hit@3':>8}{'R@5':>8}{'MRR@10':>9}{'p50 ms':>9}")
    for name, fn in CONFIGS.items():
        wrapped, latencies = timed(fn)
        report = evaluate_retriever(wrapped, gold=gold)
        row = {
            "hit@1": report.mean_hit_rate[1],
            "hit@3": report.mean_hit_rate[3],
            "recall@5": report.mean_recall[5],
            "mrr@10": mrr_at_10(report),
            "p50_ms": statistics.median(latencies),
            "misses_at_1": [q.query for q in report.per_query if q.k == 1 and not q.hit],
        }
        results[name] = row
        print(
            f"{name:<20}{row['hit@1']:>8.3f}{row['hit@3']:>8.3f}{row['recall@5']:>8.3f}"
            f"{row['mrr@10']:>9.3f}{row['p50_ms']:>9.1f}"
        )

    # İstemci gecikmesi Docker Desktop (Windows) ağ katmanını da içeriyor; sunucunun
    # kendi ölçtüğü `took` süresini ayrıca raporla.
    es = es_index.get_client()
    took = {"bm25_f5": [], "knn": []}
    for row in gold:
        took["bm25_f5"].append(es.search(index=es_index.INDEX_NAME, **es_index.bm25_query(row["query"]))["took"])
        vector = es_index.embed_query(row["query"])
        took["knn"].append(es.search(index=es_index.INDEX_NAME, **es_index.knn_query(vector))["took"])
    server_ms = {name: statistics.median(vals) for name, vals in took.items()}
    print(f"ES sunucu tarafı took p50 (ms): {server_ms}")

    OUT.write_text(
        json.dumps({"n_queries": len(gold), "results": results, "es_server_took_p50_ms": server_ms}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"yazıldı: {OUT}")


if __name__ == "__main__":
    main()
