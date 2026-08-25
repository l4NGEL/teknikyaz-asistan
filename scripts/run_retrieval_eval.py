"""Gold etiketlerle Precision@k / Recall@k hesapla.

Üretim indeksi (Chroma + BM25 + rerank) gerekir:
    python -m src.data_prep.chunking
    python -c "from src.rag.vector_store import index_chunks; from src.data_prep.chunking import chunk_corpus; index_chunks([c.to_dict() for c in chunk_corpus()])"
    python scripts/run_retrieval_eval.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.rag.eval import evaluate_retriever  # noqa: E402
from src.rag.retriever import retrieve  # noqa: E402


def main() -> None:
    def retrieve_fn(query: str, final_k: int = 10, **_kwargs):
        return retrieve(query, final_k=final_k, hierarchical=False)

    report = evaluate_retriever(retrieve_fn)
    print(report.summary_table())
    out = ROOT / "data" / "eval" / "last_retrieval_report.json"
    out.write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"yazıldı: {out}")


if __name__ == "__main__":
    main()
