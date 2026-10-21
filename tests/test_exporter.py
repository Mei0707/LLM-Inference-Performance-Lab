from prometheus_client import generate_latest

from inference_lab.exporter import BenchmarkExporter


def test_client_failures_and_timing_are_exported():
    exporter = BenchmarkExporter()
    for status, error in [
        ("success", None),
        ("timeout", "timeout"),
        ("dropped", "client_backlog_full"),
        ("canceled", "not_dispatched"),
    ]:
        exporter.observe(
            "baseline",
            {"status": status, "error_type": error, "schedule_latency_ms": 20, "ttft_ms": 5},
        )
    body = generate_latest(exporter.registry).decode()
    assert (
        'inference_lab_requests_total{error_type="timeout",server="baseline",status="timeout"} 1.0'
        in body
    )
    assert 'status="dropped"} 1.0' in body
    assert 'status="canceled"} 1.0' in body
    assert 'inference_lab_schedule_latency_seconds_count{server="baseline"} 1.0' in body
    assert 'inference_lab_ttft_seconds_sum{server="baseline"} 0.005' in body
    exporter.close()
