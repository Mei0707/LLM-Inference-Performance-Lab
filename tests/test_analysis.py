import copy

import pytest

from inference_lab.analysis import Policy, compare, regression, repetition_summary


def summary():
    return {
        "complete": True,
        "comparability": {
            "hardware": "gpu-1",
            "workload_sha256": "abc",
            "workload": {"requests": 10},
            "model_id": "test",
            "model_revision": "a" * 40,
            "tokenizer_id": "test",
            "tokenizer_revision": "a" * 40,
            "simulation": False,
            "benchmark_source_sha256": "b" * 64,
            "client_python": "3.11",
            "client_libraries": {"httpx": "test"},
            "measurement_policy": {"p99_min_samples": 100},
        },
        "requests_per_second": 5,
        "output_tokens_per_second": 100,
        "p95_ttft_ms": 10,
        "p95_latency_ms": 20,
        "error_rate": 0,
        "failed_requests": 0,
    }


@pytest.mark.parametrize("value,status", [(99, "PASS"), (90, "FAIL"), (None, "INCONCLUSIVE")])
def test_regression_states(value, status):
    a, b = summary(), summary()
    b["output_tokens_per_second"] = value
    p = Policy.model_validate(
        {"rules": [{"metric": "output_tokens_per_second", "minimum_ratio": 0.95}]}
    )
    assert regression(a, b, p)["status"] == status


def test_incompatible_never_improvement():
    a, b = summary(), summary()
    b["comparability"]["workload_sha256"] = "changed"
    b["output_tokens_per_second"] = 200
    result = compare(a, b)
    assert not result["comparable"]
    assert result["metrics"]["output_tokens_per_second"]["ratio"] is None


@pytest.mark.parametrize(
    "key,value", [("complete", False), ("comparability", {}), ("comparability", {"hardware": None})]
)
def test_missing_provenance_inconclusive(key, value):
    a = summary()
    a[key] = value
    p = Policy.model_validate({"rules": [{"metric": "error_rate", "maximum": 0.01}]})
    assert regression(a, summary(), p)["status"] == "INCONCLUSIVE"


def test_zero_baseline_ratio_inconclusive_but_absolute_is_valid():
    p = Policy.model_validate({"rules": [{"metric": "error_rate", "maximum_ratio": 1.1}]})
    assert regression(summary(), summary(), p)["status"] == "INCONCLUSIVE"
    p = Policy.model_validate({"rules": [{"metric": "error_rate", "maximum": 0.01}]})
    assert regression(summary(), summary(), p)["status"] == "PASS"


def test_partial_provenance_cannot_pass():
    baseline = summary()
    del baseline["comparability"]["workload"]
    assert not compare(baseline, baseline)["comparable"]


def test_repetitions_exclude_incomplete(config):
    m = {
        "server": config.servers[0].model_dump(),
        "workload": config.workload.model_dump(),
        "model": config.model.model_dump(),
    }
    a, b, c = summary(), summary(), summary()
    b["requests_per_second"] = 7
    c.update(complete=False, requests_per_second=100)
    groups = repetition_summary([(copy.deepcopy(m), s) for s in [a, b, c]])
    assert len(groups) == 1
    assert groups[0]["metrics"]["requests_per_second"] == {
        "median": 6,
        "min": 5,
        "max": 7,
        "repetitions_with_data": 2,
    }
