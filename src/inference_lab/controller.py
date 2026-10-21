"""Experiment lifecycle, readiness, provenance and recoverable run artifacts."""

import asyncio
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

import httpx

from inference_lab.exporter import BenchmarkExporter
from inference_lab.metrics import summarize
from inference_lab.reporting import experiment_report, run_report
from inference_lab.scheduler import execute, prepare_results
from inference_lab.storage import Journal, write_json, write_parquet
from inference_lab.telemetry import Telemetry
from inference_lab.workload import generate, load_tokenizer, workload_hash


def environment():
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
            ).stdout
        )
    except (subprocess.SubprocessError, FileNotFoundError):
        commit, dirty = None, None
    versions = {}
    for name in ["inference-lab", "httpx", "pydantic", "pyarrow", "transformers", "torch", "vllm"]:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    digest = hashlib.sha256()
    source_root = Path(__file__).resolve().parent
    for path in sorted(source_root.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "source_git_commit": commit,
        "source_git_dirty": dirty,
        "benchmark_source_sha256": digest.hexdigest(),
        "client_package_versions": versions,
    }


def headers(server):
    if not server.api_key_env:
        return {}
    key = os.environ.get(server.api_key_env)
    if not key:
        raise ValueError(f"API key environment variable {server.api_key_env} is unset")
    return {"Authorization": f"Bearer {key}"}


async def readiness(client, server, model):
    deadline = time.perf_counter() + server.readiness_timeout_seconds
    last_error = "Not ready"
    while time.perf_counter() < deadline:
        try:
            remaining = max(0.01, deadline - time.perf_counter())
            async with asyncio.timeout(remaining):
                response = await client.get("/health", timeout=min(5, remaining))
                response.raise_for_status()
                models_response = await client.get("/v1/models", timeout=min(5, remaining))
                models_response.raise_for_status()
                models = models_response.json()
                if model.id not in [entry.get("id") for entry in models.get("data", [])]:
                    raise ValueError(f"Requested model {model.id!r} not listed by server")
                selected = next(entry for entry in models["data"] if entry.get("id") == model.id)
                observed_context = selected.get("max_model_len")
                if (
                    model.tokenizer != "mock"
                    and observed_context is not None
                    and observed_context != model.max_model_len
                ):
                    raise ValueError(
                        f"Server context {observed_context} differs from declared context {model.max_model_len}"
                    )
                version_response = await client.get("/version", timeout=min(5, remaining))
                version = version_response.json() if version_response.is_success else None
                is_mock = (
                    isinstance(version, dict) and version.get("version") == "inference-lab-mock"
                )
                if (model.tokenizer == "mock") != is_mock:
                    raise ValueError(
                        "Mock tokenization must only be paired with the inference-lab mock server"
                    )
                return {"models": models, "version": version}
        except (httpx.HTTPError, ValueError, TimeoutError) as exc:
            last_error = str(exc)
            await asyncio.sleep(min(0.2, max(0, deadline - time.perf_counter())))
    raise RuntimeError(f"Readiness deadline exceeded: {last_error}")


def comparability(manifest):
    model = manifest["model"]
    server = manifest["server"]
    runtime = server["runtime"]
    gpus = manifest.get("observed_gpu", [])
    hardware = [
        {k: gpu.get(k) for k in ["uuid", "name", "memory_total_bytes", "driver_version"]}
        for gpu in gpus
    ]
    if not hardware and runtime.get("hardware_id"):
        hardware = {
            k: runtime.get(k)
            for k in ["hardware_id", "gpu_model", "gpu_memory_bytes", "driver_version"]
        }
    if manifest["simulation"]:
        hardware = "mock-protocol-fixture"
    return {
        "model_id": model["id"],
        "model_revision": model["revision"],
        "tokenizer_id": model["tokenizer_id"] or model["id"],
        "tokenizer_revision": model["tokenizer_revision"] or model["revision"],
        "workload": manifest["workload"],
        "workload_sha256": manifest["workload_sha256"],
        "hardware": hardware or None,
        "simulation": manifest["simulation"],
        "benchmark_source_sha256": manifest["environment"]["benchmark_source_sha256"],
        "client_python": manifest["environment"]["python"],
        "client_libraries": {
            name: manifest["environment"]["client_package_versions"].get(name)
            for name in ["httpx", "pyarrow", "transformers"]
        },
        "measurement_policy": manifest["report_config"],
    }


async def run_one(
    config, server, workload, repetition, destination, tokenizer, transport=None, exporter=None
):
    model = server.model or config.model
    run_id = f"{server.name}-{uuid.uuid4().hex[:12]}"
    directory = destination / run_id
    directory.mkdir(parents=True)
    prompts = generate(workload, tokenizer, config.experiment.random_seed)
    write_json(directory / "workload.json", [p.to_dict() for p in prompts])
    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "experiment_name": config.experiment.name,
        "started_at": datetime.now(UTC).isoformat(),
        "state": "initializing",
        "simulation": model.tokenizer == "mock",
        "repetition": repetition,
        "seed": config.experiment.random_seed,
        "model": model.model_dump(),
        "server": server.model_dump(),
        "workload": workload.model_dump(),
        "report_config": config.report.model_dump(),
        "metrics_config": config.metrics.model_dump(),
        "workload_sha256": workload_hash(prompts),
        "environment": environment(),
        "runtime_metadata_provenance": "server.runtime is operator-supplied; environment packages are client-local; observed_server and observed_gpu are discovered",
    }
    write_json(directory / "manifest.json", manifest)
    journal = Journal(directory / "requests.jsonl")

    def observe(row):
        journal.append(row)
        if exporter:
            exporter.observe(server.name, row)

    results = prepare_results(prompts, workload, config.experiment.random_seed, run_id)
    telemetry = None
    measurement_start = None
    duration = 0.0
    fatal = None
    interrupted = False
    try:
        limits = httpx.Limits(
            max_connections=workload.concurrency + 4,
            max_keepalive_connections=workload.concurrency + 4,
        )
        async with httpx.AsyncClient(
            base_url=server.base_url,
            headers=headers(server),
            limits=limits,
            timeout=server.request_timeout_seconds,
            transport=transport,
            trust_env=False,
        ) as client:
            manifest["observed_server"] = await readiness(client, server, model)
            if config.experiment.warmup_requests:
                warmup = workload.model_copy(
                    update={
                        "mode": "closed_loop",
                        "requests": config.experiment.warmup_requests,
                        "concurrency": min(workload.concurrency, config.experiment.warmup_requests),
                        "arrival_rate": None,
                    }
                )
                warm_prompts = generate(warmup, tokenizer, config.experiment.random_seed + 1)
                warm_results = prepare_results(
                    warm_prompts, warmup, config.experiment.random_seed + 1, run_id + "-warmup"
                )
                async with asyncio.timeout(config.experiment.timeout_seconds):
                    await execute(
                        client,
                        server,
                        model,
                        warmup,
                        warm_prompts,
                        warm_results,
                        time.perf_counter(),
                        lambda _: None,
                    )
                write_json(directory / "warmup.json", [r.to_dict() for r in warm_results])
                if not any(r.status == "success" for r in warm_results):
                    raise RuntimeError("All warm-up requests failed")
            telemetry = Telemetry(config.metrics, client, time.perf_counter())
            await telemetry.start()
            manifest["observed_gpu"] = telemetry.gpu_metadata
            manifest["state"] = "running"
            write_json(directory / "manifest.json", manifest)
            measurement_start = time.perf_counter()
            telemetry.origin = measurement_start
            try:
                async with asyncio.timeout(config.experiment.timeout_seconds):
                    await execute(
                        client,
                        server,
                        model,
                        workload,
                        prompts,
                        results,
                        measurement_start,
                        observe,
                    )
                manifest["state"] = "completed"
            finally:
                duration = time.perf_counter() - measurement_start
                await telemetry.stop()
            if config.metrics.require_gpu and not any(
                row.get("memory_used_bytes") is not None
                and row.get("utilization_percent") is not None
                for row in telemetry.gpu_rows
            ):
                raise RuntimeError(
                    "GPU telemetry was required but no usable utilization/memory samples were collected"
                )
    except asyncio.CancelledError:
        manifest["state"] = "interrupted"
        fatal = "Benchmark interrupted"
        interrupted = True
    except Exception as exc:
        manifest["state"] = "incomplete"
        fatal = f"{type(exc).__name__}: {exc}"
    finally:
        if measurement_start is None:
            for result in results:
                result.status, result.error_type = "canceled", "environment_not_ready"
                observe(result.to_dict())
        journal.close()
        manifest["finished_at"] = datetime.now(UTC).isoformat()
        manifest["failure_reason"] = fatal
        manifest["measurement_duration_seconds"] = duration
        rows = [r.to_dict() for r in results]
        summary = summarize(rows, duration, config.report)
        summary.update(
            {
                "run_id": run_id,
                "complete": manifest["state"] == "completed",
                "simulation": manifest["simulation"],
                "comparability": comparability(manifest),
                "telemetry": telemetry.summary() if telemetry else {},
                "report_config": config.report.model_dump(),
            }
        )
        manifest["telemetry_errors"] = telemetry.errors if telemetry else []
        write_parquet(directory / "requests.parquet", rows)
        write_parquet(directory / "gpu_metrics.parquet", telemetry.gpu_rows if telemetry else [])
        write_parquet(
            directory / "server_metrics.parquet", telemetry.server_rows if telemetry else []
        )
        write_json(directory / "manifest.json", manifest)
        write_json(directory / "summary.json", summary)
        run_report(directory, manifest, summary, rows)
    if interrupted:
        raise asyncio.CancelledError
    return directory


async def run_experiment(config, output="results", transport=None):
    destination = Path(output) / f"{config.experiment.name}-{uuid.uuid4().hex[:10]}"
    destination.mkdir(parents=True)
    write_json(destination / "config.json", config.model_dump())
    tokenizers = {}
    directories = []
    exporter = BenchmarkExporter() if config.metrics.exporter_port else None
    try:
        if exporter:
            exporter.start(config.metrics.exporter_port, config.metrics.exporter_host)
        for server, workload, repetition in config.runs():
            model = server.model or config.model
            key = json.dumps(model.model_dump(), sort_keys=True)
            if key not in tokenizers:
                tokenizers[key] = await asyncio.to_thread(load_tokenizer, model)
            print(
                f"Running {server.name}, concurrency={workload.concurrency}, input={workload.input_tokens}, repetition={repetition + 1}",
                flush=True,
            )
            directory = await run_one(
                config,
                server,
                workload,
                repetition,
                destination,
                tokenizers[key],
                transport,
                exporter,
            )
            directories.append(directory)
    finally:
        try:
            # Also discover a run finalized during a cancellation.
            directories = sorted(p.parent for p in destination.glob("*/summary.json"))
            if directories:
                experiment_report(destination, directories)
        finally:
            if exporter:
                await asyncio.to_thread(exporter.close)
    return destination
