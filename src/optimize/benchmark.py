"""Farklı model varyantlarını (base / quantized / pruned / distilled) tutarlı biçimde
karşılaştırmak için ortak bir benchmark aracı: gecikme (latency), verim (throughput),
bellek kullanımı ve (varsa) doğruluk.

Bu modül LLMOps'un "A/B test" bileşenini (src/llmops/ab_test.py) besler: buradan çıkan
metrikler MLflow'a loglanır.
"""
from __future__ import annotations

import os
import time
from dataclasses import asdict, dataclass

import psutil
import torch


@dataclass
class BenchmarkResult:
    name: str
    avg_latency_ms: float
    p95_latency_ms: float
    throughput_samples_per_sec: float
    peak_memory_mb: float | None
    cpu_memory_mb: float
    num_runs: int

    def to_dict(self) -> dict:
        return asdict(self)


def benchmark_inference(
    predict_fn,
    inputs: list,
    name: str = "model",
    warmup_runs: int = 3,
    timed_runs: int = 20,
) -> BenchmarkResult:
    """predict_fn(input) -> herhangi bir çıktı üreten çağrılabilir bir fonksiyon."""
    device_available = torch.cuda.is_available()
    if device_available:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    for i in range(warmup_runs):
        predict_fn(inputs[i % len(inputs)])
    if device_available:
        torch.cuda.synchronize()

    latencies = []
    for i in range(timed_runs):
        start = time.perf_counter()
        predict_fn(inputs[i % len(inputs)])
        if device_available:
            torch.cuda.synchronize()
        latencies.append((time.perf_counter() - start) * 1000)

    latencies.sort()
    avg_latency = sum(latencies) / len(latencies)
    p95_latency = latencies[int(len(latencies) * 0.95) - 1]
    throughput = 1000.0 / avg_latency if avg_latency > 0 else 0.0
    peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2) if device_available else None
    # CUDA "mevcut mu" (torch.cuda.is_available()) ile "BU model GPU'da mi" AYNI SEY
    # DEGIL -- predict_fn kapali bir fonksiyon oldugu icin modelin hangi cihazda
    # oldugunu buradan bilemeyiz. Bu yuzden CPU RSS'i HER ZAMAN da olcuyoruz: model
    # CPU'daysa (orn. .to("cuda") hic cagrilmamissa) peak_memory_mb, o an baska bir
    # seyin (orn. daha once yuklenmis, hic serbest birakilmamis buyuk bir GPU modelinin)
    # GPU'da sabit durmasini yansitip modeller arasi farksiz cikabilir; cpu_memory_mb
    # bu durumda gercek, ayirt edici sinyali verir.
    cpu_memory = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)

    return BenchmarkResult(
        name=name,
        avg_latency_ms=avg_latency,
        p95_latency_ms=p95_latency,
        throughput_samples_per_sec=throughput,
        peak_memory_mb=peak_memory,
        cpu_memory_mb=cpu_memory,
        num_runs=timed_runs,
    )


def compare(results: list[BenchmarkResult]) -> str:
    lines = [
        f"{'Model':<25}{'Ort. Gecikme (ms)':<20}{'p95 (ms)':<15}"
        f"{'GPU Bellek (MB)':<18}{'CPU Bellek (MB)':<18}"
    ]
    for r in results:
        gpu_mem = f"{r.peak_memory_mb:.1f}" if r.peak_memory_mb is not None else "N/A"
        lines.append(
            f"{r.name:<25}{r.avg_latency_ms:<20.2f}{r.p95_latency_ms:<15.2f}"
            f"{gpu_mem:<18}{r.cpu_memory_mb:<18.1f}"
        )
    return "\n".join(lines)
