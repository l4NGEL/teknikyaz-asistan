"""vLLM uç noktasında p50 / p95 / p99 gecikme ölçümü.

`benchmark_inference` ile aynı BenchmarkResult şemasını üretir; ek olarak p50/p99
döner çünkü Diagnis ilanı P95 hedefini açıkça istiyor.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

from src.optimize.benchmark import BenchmarkResult, benchmark_inference
from src.optimize.vllm_client import generate_vllm


@dataclass
class VllmBenchmarkResult:
    inner: BenchmarkResult
    p50_latency_ms: float
    p99_latency_ms: float

    def to_dict(self) -> dict:
        payload = asdict(self.inner)
        payload["p50_latency_ms"] = self.p50_latency_ms
        payload["p99_latency_ms"] = self.p99_latency_ms
        return payload


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    rank = (len(xs) - 1) * (p / 100.0)
    low = int(rank)
    high = min(low + 1, len(xs) - 1)
    weight = rank - low
    return xs[low] + (xs[high] - xs[low]) * weight


def benchmark_vllm(
    prompts: list[str] | None = None,
    *,
    name: str = "vllm",
    warmup_runs: int = 2,
    timed_runs: int = 20,
    **generate_kwargs,
) -> VllmBenchmarkResult:
    prompts = prompts or [
        "Bir README'de hangi bölümler olmalı?",
        "API dokümantasyonu nasıl yapılandırılır?",
        "Kurulum kılavuzunda hangi adımlar bulunur?",
    ]
    latencies: list[float] = []

    def predict(prompt: str) -> str:
        import time

        start = time.perf_counter()
        text = generate_vllm(prompt, **generate_kwargs)
        latencies.append((time.perf_counter() - start) * 1000)
        return text

    inner = benchmark_inference(
        predict,
        inputs=prompts,
        name=name,
        warmup_runs=warmup_runs,
        timed_runs=timed_runs,
    )
    # warmup da predict'i çağırdığı için ilk warmup_runs örneği düşür
    timed = latencies[warmup_runs:] if len(latencies) > warmup_runs else latencies
    return VllmBenchmarkResult(
        inner=inner,
        p50_latency_ms=percentile(timed, 50),
        p99_latency_ms=percentile(timed, 99),
    )
