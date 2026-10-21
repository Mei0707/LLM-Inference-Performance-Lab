"""CPU-only streaming protocol fixture, deliberately identified as a simulation."""

import asyncio
import json
import time

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response, StreamingResponse
from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest
from pydantic import BaseModel, Field


class Completion(BaseModel):
    model: str
    prompt: list[int] = Field(min_length=1)
    max_tokens: int = Field(gt=0)
    stream: bool = True


def create_mock_app(delay=0.003, failure=None, include_usage=True):
    app = FastAPI(title="Inference Lab MOCK protocol fixture")
    registry = CollectorRegistry()
    running = Gauge("vllm:num_requests_running", "Mock active streams", registry=registry)
    prompt_tokens = Counter("vllm:prompt_tokens", "Mock input tokens", registry=registry)
    generation_tokens = Counter("vllm:generation_tokens", "Mock output tokens", registry=registry)
    successes = Counter("vllm:request_success", "Mock completed requests", registry=registry)
    errors = Counter("inference_lab_mock_failures", "Mock failures", registry=registry)
    ttft = Histogram(
        "vllm:time_to_first_token_seconds", "Mock first-content latency", registry=registry
    )
    latency = Histogram(
        "vllm:e2e_request_latency_seconds", "Mock service latency", registry=registry
    )

    @app.get("/health")
    async def health():
        return {"status": "ok", "simulation": True}

    @app.get("/version")
    async def version():
        return {"version": "inference-lab-mock", "simulation": True}

    @app.get("/v1/models")
    async def models():
        return {
            "object": "list",
            "data": [{"id": "mock-model", "object": "model", "max_model_len": 8192}],
        }

    @app.get("/metrics")
    async def metrics():
        return Response(generate_latest(registry), media_type="text/plain; version=0.0.4")

    @app.post("/v1/completions")
    async def completions(body: Completion):
        if body.model != "mock-model":
            raise HTTPException(404, "Unknown mock model")
        if failure == "rate_limit":
            errors.inc()
            raise HTTPException(429, "Synthetic rate limit")
        if failure == "oom":
            errors.inc()
            raise HTTPException(500, "Synthetic CUDA out of memory")

        async def stream():
            running.inc()
            started = time.perf_counter()
            total = min(body.max_tokens, 8)
            prompt_tokens.inc(len(body.prompt))

            def event(value):
                return f"data: {json.dumps(value)}\n\n"

            try:
                yield ": mock stream\n\n"
                yield event({"choices": [{"delta": {"role": "assistant"}}]})
                if failure == "timeout":
                    await asyncio.sleep(10)
                if failure == "invalid":
                    yield "data: {broken\n\n"
                    return
                if failure != "empty":
                    for start in range(0, total, 2):
                        await asyncio.sleep(delay)
                        if start == 0:
                            ttft.observe(time.perf_counter() - started)
                        count = min(2, total - start)
                        generation_tokens.inc(count)
                        yield event({"choices": [{"index": 0, "text": " token" * count}]})
                yield event(
                    {
                        "choices": [
                            {
                                "text": "",
                                "finish_reason": "length" if total == body.max_tokens else "stop",
                            }
                        ]
                    }
                )
                if include_usage:
                    yield event(
                        {
                            "choices": [],
                            "usage": {
                                "prompt_tokens": len(body.prompt),
                                "completion_tokens": 0 if failure == "empty" else total,
                            },
                        }
                    )
                if failure != "incomplete":
                    yield "data: [DONE]\n\n"
                    successes.inc()
            finally:
                running.dec()
                latency.observe(time.perf_counter() - started)

        return StreamingResponse(stream(), media_type="text/event-stream")

    return app


app = create_mock_app()
