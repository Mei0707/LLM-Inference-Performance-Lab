import asyncio
import json

import httpx
import pytest

from inference_lab.controller import run_experiment
from inference_lab.mock_server import create_mock_app
from inference_lab.reporting import regenerate
from inference_lab.storage import read_parquet


async def test_entire_pipeline_and_regeneration(config, tmp_path):
    config.metrics.collect_server = True
    config.experiment.warmup_requests = 2
    directory = await run_experiment(
        config, tmp_path, httpx.ASGITransport(app=create_mock_app(delay=0))
    )
    run = next(directory.glob("*/manifest.json")).parent
    summary = json.loads((run / "summary.json").read_text())
    assert summary["complete"]
    assert summary["successful_requests"] == 6
    assert summary["simulation"]
    assert summary["telemetry"]["server_samples"] > 0
    assert len(read_parquet(run / "requests.parquet")) == 6
    assert read_parquet(run / "gpu_metrics.parquet") == []
    assert len((run / "requests.jsonl").read_text().splitlines()) == 6
    assert "SIMULATION" in (run / "report.md").read_text()
    assert (run / "latency.png").stat().st_size > 1000
    regenerate(directory)
    assert json.loads((run / "summary.json").read_text()) == summary
    assert (directory / "aggregate.json").exists()
    (run / "summary.json").unlink()
    regenerate(directory)
    assert json.loads((run / "summary.json").read_text()) == summary


@pytest.mark.parametrize(
    "failure,error",
    [
        ("rate_limit", "rate_limit"),
        ("oom", "out_of_memory"),
        ("empty", "empty_response"),
        ("incomplete", "incomplete_stream"),
        ("invalid", "invalid_response"),
    ],
)
async def test_failed_requests_survive(config, tmp_path, failure, error):
    directory = await run_experiment(
        config, tmp_path, httpx.ASGITransport(app=create_mock_app(failure=failure, delay=0))
    )
    summary = json.loads(next(directory.glob("*/summary.json")).read_text())
    assert summary["requests"] == 6
    assert summary["failed_requests"] == 6
    assert summary["error_rate"] == 1
    assert summary["error_counts"][error] == 6


async def test_experiment_timeout_finalizes_canceled(config, tmp_path):
    config.experiment.timeout_seconds = 0.03
    directory = await run_experiment(
        config, tmp_path, httpx.ASGITransport(app=create_mock_app(delay=1))
    )
    run = next(directory.glob("*/summary.json")).parent
    summary = json.loads((run / "summary.json").read_text())
    assert not summary["complete"]
    assert summary["failed_requests"] == 6
    assert len(read_parquet(run / "requests.parquet")) == 6
    assert all(r["status"] == "canceled" for r in read_parquet(run / "requests.parquet"))


async def test_external_cancellation_preserves_reports(config, tmp_path):
    task = asyncio.create_task(
        run_experiment(config, tmp_path, httpx.ASGITransport(app=create_mock_app(delay=1)))
    )
    # Wait until actual inference has started rather than depending on setup speed.
    for _ in range(500):
        manifests = list(tmp_path.glob("*/*/manifest.json"))
        if manifests and json.loads(manifests[0].read_text())["state"] == "running":
            break
        await asyncio.sleep(0.002)
    else:
        pytest.fail("Run never reached the measured phase")
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    manifest = json.loads(manifests[0].read_text())
    assert manifest["state"] == "interrupted"
    assert (manifests[0].parent / "report.md").exists()
    assert len(read_parquet(manifests[0].parent / "requests.parquet")) == 6


async def test_readiness_failure_is_reported(config, tmp_path):
    config.servers[0].readiness_timeout_seconds = 0.02

    async def handler(request):
        return httpx.Response(503)

    directory = await run_experiment(config, tmp_path, httpx.MockTransport(handler))
    summary = json.loads(next(directory.glob("*/summary.json")).read_text())
    assert not summary["complete"]
    assert summary["error_counts"] == {"environment_not_ready": 6}
    assert summary["duration_seconds"] == 0


async def test_missing_usage_visible_end_to_end(config, tmp_path):
    directory = await run_experiment(
        config, tmp_path, httpx.ASGITransport(app=create_mock_app(include_usage=False, delay=0))
    )
    summary = json.loads(next(directory.glob("*/summary.json")).read_text())
    assert summary["successful_requests"] == 6
    assert summary["token_usage_coverage"] == 0
    assert summary["output_tokens_per_second"] is None


async def test_required_gpu_sample_failure_is_incomplete(config, tmp_path, monkeypatch):
    class BrokenSampler:
        def __init__(self, indices):
            pass

        def start(self):
            return [{"name": "fake-test-device"}]

        def sample(self, timestamp):
            raise RuntimeError("device lost")

        def close(self):
            pass

    monkeypatch.setattr("inference_lab.telemetry.GPUMonitor", BrokenSampler)
    config.metrics.collect_gpu = config.metrics.require_gpu = True
    directory = await run_experiment(
        config, tmp_path, httpx.ASGITransport(app=create_mock_app(delay=0))
    )
    summary = json.loads(next(directory.glob("*/summary.json")).read_text())
    assert not summary["complete"]
    assert summary["successful_requests"] == 6
    assert summary["telemetry"]["gpu_samples"] == 0
    assert summary["telemetry"]["errors"][0]["source"] == "gpu"
