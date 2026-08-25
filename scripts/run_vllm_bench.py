"""vLLM OpenAI uyumlu sunucuya karşı p50/p95/p99 ölç.

    docker compose --profile gpu -f docker/docker-compose.yml up vllm
    python scripts/run_vllm_bench.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.optimize.vllm_benchmark import benchmark_vllm  # noqa: E402


def main() -> None:
    result = benchmark_vllm(warmup_runs=1, timed_runs=10)
    payload = result.to_dict()
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"p50={result.p50_latency_ms:.1f} ms  p95={result.inner.p95_latency_ms:.1f} ms  p99={result.p99_latency_ms:.1f} ms")


if __name__ == "__main__":
    main()
