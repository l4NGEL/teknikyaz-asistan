import time

from src.optimize.benchmark import benchmark_inference, compare


def _fast_predict(x):
    return x * 2


def _slow_predict(x):
    time.sleep(0.005)
    return x * 2


def test_benchmark_reports_positive_latency_and_throughput():
    result = benchmark_inference(_fast_predict, inputs=list(range(5)), name="fast", warmup_runs=2, timed_runs=5)

    assert result.name == "fast"
    assert result.avg_latency_ms >= 0
    assert result.throughput_samples_per_sec > 0
    assert result.num_runs == 5


def test_slower_function_has_higher_latency():
    fast = benchmark_inference(_fast_predict, inputs=list(range(5)), name="fast", warmup_runs=2, timed_runs=10)
    slow = benchmark_inference(_slow_predict, inputs=list(range(5)), name="slow", warmup_runs=2, timed_runs=10)

    assert slow.avg_latency_ms > fast.avg_latency_ms


def test_compare_output_contains_all_model_names():
    fast = benchmark_inference(_fast_predict, inputs=[1], name="fast", warmup_runs=1, timed_runs=3)
    slow = benchmark_inference(_slow_predict, inputs=[1], name="slow", warmup_runs=1, timed_runs=3)

    report = compare([fast, slow])

    assert "fast" in report
    assert "slow" in report
