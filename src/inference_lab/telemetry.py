"""Preserve Prometheus samples by name/labels without depending on vLLM metric aliases."""

import asyncio
import json
import math

from prometheus_client.parser import text_string_to_metric_families

from inference_lab.client import elapsed_ms
from inference_lab.gpu_monitor import GPUMonitor


def parse_prometheus(body, timestamp_ms):
    return [
        {
            "timestamp_ms": timestamp_ms,
            "name": sample.name,
            "labels": json.dumps(sample.labels, sort_keys=True),
            "value": sample.value,
        }
        for family in text_string_to_metric_families(body)
        for sample in family.samples
        if math.isfinite(sample.value)
    ]


class Telemetry:
    def __init__(self, config, client, origin):
        self.config, self.client, self.origin = config, client, origin
        self.gpu_rows, self.server_rows, self.errors = [], [], []
        self.gpu = None
        self.gpu_metadata = []
        self.stop_event = asyncio.Event()
        self.task = None

    def error(self, source, exc):
        message = f"{type(exc).__name__}: {exc}"
        if not any(x["source"] == source and x["error"] == message for x in self.errors):
            self.errors.append(
                {"source": source, "error": message, "timestamp_ms": elapsed_ms(self.origin)}
            )

    async def start(self):
        if self.config.collect_gpu:
            self.gpu = GPUMonitor(self.config.gpu_indices)
            try:
                self.gpu_metadata = await asyncio.to_thread(self.gpu.start)
            except Exception as exc:
                self.error("gpu", exc)
                if self.config.require_gpu:
                    self.gpu.close()
                    raise RuntimeError("GPU telemetry is required but NVML is unavailable") from exc
                self.gpu.close()
                self.gpu = None
        self.task = asyncio.create_task(self.loop())

    async def sample(self):
        timestamp = elapsed_ms(self.origin)
        if self.gpu:
            try:
                self.gpu_rows.extend(await asyncio.to_thread(self.gpu.sample, timestamp))
            except Exception as exc:
                self.error("gpu", exc)
        if self.config.collect_server:
            try:
                response = await self.client.get("/metrics", timeout=2)
                response.raise_for_status()
                self.server_rows.extend(parse_prometheus(response.text, timestamp))
            except Exception as exc:
                self.error("server", exc)

    async def loop(self):
        while not self.stop_event.is_set():
            await self.sample()
            try:
                await asyncio.wait_for(
                    self.stop_event.wait(), self.config.sample_interval_ms / 1000
                )
            except TimeoutError:
                pass

    async def stop(self):
        self.stop_event.set()
        if self.task:
            await self.task
        if self.gpu:
            await asyncio.to_thread(self.gpu.close)

    def summary(self):
        return summarize_telemetry(self.gpu_rows, self.server_rows, self.errors)


def summarize_telemetry(gpu_rows, server_rows, errors):
    def mean(rows, key):
        values = [r[key] for r in rows if r.get(key) is not None]
        return sum(values) / len(values) if values else None

    return {
        "gpu_samples": len(gpu_rows),
        "server_samples": len(server_rows),
        "gpu_utilization_percent_mean": mean(gpu_rows, "utilization_percent"),
        "gpu_memory_used_bytes_mean": mean(gpu_rows, "memory_used_bytes"),
        "gpu_memory_used_bytes_peak": max(
            (r["memory_used_bytes"] for r in gpu_rows if r.get("memory_used_bytes") is not None),
            default=None,
        ),
        "kv_cache_usage_fraction_mean": mean(
            [
                {"value": r["value"]}
                for r in server_rows
                if r["name"] in {"vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc"}
            ],
            "value",
        ),
        "server_metric_names": sorted({r["name"] for r in server_rows}),
        "errors": errors,
    }
