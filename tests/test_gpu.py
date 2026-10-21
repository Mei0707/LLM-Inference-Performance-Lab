"""Explicit opt-in; never silently launch paid infrastructure or model downloads in CI."""

import json
import os

import pytest

from inference_lab.config import load_config
from inference_lab.controller import run_experiment


@pytest.mark.gpu
async def test_real_gpu_pipeline(tmp_path):
    path = os.environ.get("INFERENCE_LAB_GPU_CONFIG")
    if not path:
        pytest.skip("Set INFERENCE_LAB_GPU_CONFIG to a real, pinned experiment on the GPU host")
    config = load_config(path)
    assert config.model.tokenizer == "huggingface"
    config.experiment.repetitions = 1
    config.experiment.warmup_requests = 1
    config.workload.concurrency = 1
    config.workload.input_tokens = 32
    config.workload.max_output_tokens = 8
    config.workload.requests = 3
    config.servers = config.servers[:1]
    config.metrics.collect_gpu = True
    config.metrics.require_gpu = True
    directory = await run_experiment(config, tmp_path)
    summary = json.loads(next(directory.glob("*/summary.json")).read_text())
    assert summary["complete"]
    assert summary["successful_requests"] == 3
    assert summary["p50_ttft_ms"] > 0
    assert summary["telemetry"]["gpu_samples"] > 0
    assert summary["telemetry"]["server_samples"] > 0
