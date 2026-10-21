"""Synthetic figure coverage; all fabricated test observations stay in tmp_path."""

import copy
import json

import httpx

from inference_lab.controller import run_experiment
from inference_lab.mock_server import create_mock_app
from inference_lab.reporting import experiment_report
from inference_lab.storage import write_json


async def test_configuration_resource_and_scaling_charts(config, tmp_path):
    config.experiment.repetitions = 1
    config.workload.concurrency = [1, 2]
    config.workload.input_tokens = [16, 32]
    config.workload.requests = 2
    second = copy.deepcopy(config.servers[0])
    second.name = "candidate"
    config.servers.append(second)
    config.report.slo_latency_ms = 1000
    directory = await run_experiment(
        config, tmp_path, httpx.ASGITransport(app=create_mock_app(delay=0))
    )
    runs = []
    for path in directory.glob("*/summary.json"):
        summary = json.loads(path.read_text())
        summary["telemetry"].update(
            gpu_memory_used_bytes_mean=1024**3, kv_cache_usage_fraction_mean=0.4
        )
        write_json(path, summary)
        runs.append(path.parent)
    experiment_report(directory, sorted(runs))
    for name in [
        "configuration-throughput.png",
        "configuration-memory.png",
        "gpu-memory-vs-concurrency.png",
        "kv-cache-vs-tail-latency.png",
        "concurrency-vs-requests_per_second.png",
        "input_tokens-vs-p95_ttft_ms.png",
    ]:
        assert (directory / name).stat().st_size > 1000
        assert name in (directory / "report.md").read_text()
    assert "Highest observed median SLO goodput" in (directory / "report.md").read_text()
