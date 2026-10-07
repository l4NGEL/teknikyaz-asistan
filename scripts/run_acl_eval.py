"""Yetki farkındalıklı retrieval eval'i: sızıntı, kalite kaybı, gecikme maliyeti.

Aynı 32 soruluk gold set her kullanıcı rolüyle çalıştırılır ve üç şey ölçülür:
  1. leak@5: dönen belgelerden kullanıcının okuyamadığı oran (hedef: tam 0).
  2. Okuyabildiği gold belgede Hit@1 / Recall@5, ACL'siz taban çizgisine göre
     (ön-süzme + aday havuzu genişletme kaliteyi düşürüyor mu?).
  3. Okuyamadığı gold belge için doğru red: gold belge top-5'te hiç görünmemeli.
Ek olarak `acl_filter` aşamasının gecikmesi trace'lerden okunur.

Üretim indeksi gerekir (bkz. scripts/run_retrieval_eval.py):
    python scripts/run_acl_eval.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.governance.acl import default_policy, load_principals  # noqa: E402
from src.llmops import tracing  # noqa: E402
from src.rag.eval import load_gold  # noqa: E402
from src.rag.retriever import retrieve_with_report  # noqa: E402

K = 5
ROLES = ("musteri", "gelistirici", "guvenlik")
TRACE_PATH = ROOT / "data" / "logs" / "acl_eval_traces.jsonl"


def titles(hits: list[dict]) -> list[str]:
    seen: list[str] = []
    for h in hits:
        if h["doc_title"] not in seen:
            seen.append(h["doc_title"])
    return seen


def main() -> None:
    tracing.TRACES_PATH = TRACE_PATH
    TRACE_PATH.unlink(missing_ok=True)
    policy = default_policy()
    principals = load_principals()
    gold = load_gold()

    baseline = {}
    for row in gold:
        hits, _ = retrieve_with_report(row["query"], final_k=K, hierarchical=False)
        baseline[row["query"]] = titles(hits)

    report = {"k": K, "n_queries": len(gold), "roles": {}}
    for role in ROLES:
        principal = principals[role]
        returned_total = leaked = 0
        allowed_q = denied_q = 0
        hit1 = base_hit1 = recall = base_recall = 0
        correctly_denied = 0
        denied_candidates = 0
        for row in gold:
            relevant = set(row["relevant_titles"])
            hits, denied = retrieve_with_report(row["query"], final_k=K, hierarchical=False, principal=principal)
            got = titles(hits)
            denied_candidates += denied
            returned_total += len(got)
            leaked += sum(not policy.can_read(principal, t) for t in got)

            readable = {t for t in relevant if policy.can_read(principal, t)}
            if readable:
                allowed_q += 1
                hit1 += bool(got[:1]) and got[0] in readable
                recall += len(readable & set(got)) / len(readable)
                base = baseline[row["query"]]
                base_hit1 += bool(base[:1]) and base[0] in readable
                base_recall += len(readable & set(base)) / len(readable)
            else:
                denied_q += 1
                correctly_denied += not (relevant & set(got))

        report["roles"][role] = {
            "clearance": principal.clearance,
            "leak_rate": leaked / max(returned_total, 1),
            "leaked_docs": leaked,
            "queries_with_readable_gold": allowed_q,
            "hit1_acl": hit1 / max(allowed_q, 1),
            "hit1_baseline": base_hit1 / max(allowed_q, 1),
            f"recall{K}_acl": recall / max(allowed_q, 1),
            f"recall{K}_baseline": base_recall / max(allowed_q, 1),
            "queries_with_forbidden_gold": denied_q,
            "correct_denial_rate": correctly_denied / denied_q if denied_q else None,
            "denied_candidates_total": denied_candidates,
        }

    stages = tracing.stage_latency_summary(tracing.load_traces(TRACE_PATH))
    report["latency_ms"] = {name: stages[name] for name in ("acl_filter", "dense", "bm25", "rerank", "retrieve") if name in stages}

    print(f"{'rol':<12}{'leak':>6}{'Hit@1 acl/base':>17}{'R@5 acl/base':>15}{'red':>8}")
    for role, r in report["roles"].items():
        print(
            f"{role:<12}{r['leak_rate']:>6.2f}"
            f"{r['hit1_acl']:>9.3f}/{r['hit1_baseline']:.3f}"
            f"{r[f'recall{K}_acl']:>8.3f}/{r[f'recall{K}_baseline']:.3f}"
            f"{'—' if r['correct_denial_rate'] is None else format(r['correct_denial_rate'], '.2f'):>8}"
            f"  ({r['queries_with_readable_gold']} izinli, "
            f"{r['queries_with_forbidden_gold']} yasak gold)"
        )
    for name, s in report["latency_ms"].items():
        print(f"  {name:<11} p50 {s['p50_ms']:.2f} ms  p95 {s['p95_ms']:.2f} ms")

    out = ROOT / "data" / "eval" / "last_acl_report.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"yazıldı: {out}")


if __name__ == "__main__":
    main()
