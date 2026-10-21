import pytest

from inference_lab.config import Report
from inference_lab.metrics import RequestResult, percentile, summarize


def row(index, status="success", tokens=4):
    r = RequestResult(str(index), "run", "short", 0, 8, 4)
    r.status, r.output_tokens = status, tokens
    r.dispatched_at_ms, r.latency_ms = 0, 10
    r.ttft_ms, r.schedule_latency_ms = 2, 12
    return r.to_dict()


def test_percentiles_interpolate_and_ignore_missing():
    assert percentile([0, 10, None, 20, float("nan")], 95) == 19
    assert percentile([], 50) is None


def test_failure_denominator_and_tokens():
    result = summarize([row(1), row(2, "timeout", 100)], 2, Report())
    assert result["error_rate"] == 0.5
    assert result["requests_per_second"] == 0.5
    assert result["output_tokens_per_second"] == 2
    assert result["p95_latency_ms"] == 10
    assert result["p99_ttft_ms"] is None


def test_unknown_usage_not_zero():
    result = summarize([row(1), row(2, tokens=None)], 2, Report())
    assert result["token_usage_coverage"] == 0.5
    assert result["output_tokens_per_second"] is None


def test_slo_includes_failed_requests():
    result = summarize([row(1), row(2, "timeout")], 2, Report(slo_latency_ms=15))
    assert result["slo_attainment"] == 0.5
    assert result["slo_goodput_requests_per_second"] == 0.5


def test_zero_duration_and_empty():
    result = summarize([], 0, Report())
    assert result["error_rate"] is None
    assert result["requests_per_second"] is None


def test_p99_sample_requirement():
    result = summarize([row(i) for i in range(100)], 2, Report())
    assert result["p99_ttft_ms"] == pytest.approx(2)
