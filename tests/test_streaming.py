import asyncio
import json
import time

import httpx
import pytest

from inference_lab.client import request
from inference_lab.scheduler import prepare_results
from inference_lab.workload import MockTokenizer, generate


class TimedStream(httpx.AsyncByteStream):
    def __init__(self, frames):
        self.frames = frames

    async def __aiter__(self):
        for delay, body in self.frames:
            await asyncio.sleep(delay)
            # Deliberately split JSON and SSE separators across transport chunks.
            encoded = body.encode()
            mid = len(encoded) // 2
            yield encoded[:mid]
            yield encoded[mid:]


def event(data):
    return "data: " + json.dumps(data) + "\r\n\r\n"


async def measure(config, frames, status=200):
    prompt = generate(config.workload, MockTokenizer(), 42)[0]
    row = prepare_results([prompt], config.workload, 42, "test")[0]

    async def handler(req):
        assert json.loads(req.content)["prompt"] == prompt.token_ids
        return httpx.Response(status, stream=TimedStream(frames))

    async with httpx.AsyncClient(
        base_url="http://test", transport=httpx.MockTransport(handler)
    ) as client:
        await request(
            client,
            config.servers[0],
            config.model,
            config.workload,
            prompt,
            row,
            time.perf_counter(),
        )
    return row


def final_frames(input_tokens, output_tokens=4, usage=True):
    frames = [(0, event({"choices": [{"text": "", "finish_reason": "length"}]}))]
    if usage:
        frames.append(
            (
                0,
                event(
                    {
                        "choices": [],
                        "usage": {
                            "prompt_tokens": input_tokens,
                            "completion_tokens": output_tokens,
                        },
                    }
                ),
            )
        )
    return frames + [(0, "data: [DONE]\n\n")]


async def test_metadata_not_ttft_and_multitoken_chunks(config):
    frames = [
        (0, ": keepalive\n\n"),
        (0.01, event({"choices": [{"delta": {"role": "assistant"}}]})),
        (0.02, event({"choices": [{"text": "two tokens"}]})),
        (0.02, event({"choices": [{"text": " two more"}]})),
    ]
    row = await measure(config, frames + final_frames(32))
    assert row.status == "success"
    assert row.ttft_ms >= 25
    assert row.output_tokens == 4
    assert row.content_events == 2
    assert row.tpot_ms == pytest.approx((row.last_content_at_ms - row.first_content_at_ms) / 3)
    assert row.tpot_ms >= 5


async def test_one_chunk_cannot_measure_tpot(config):
    row = await measure(
        config, [(0, event({"choices": [{"text": "four tokens at once"}]}))] + final_frames(32)
    )
    assert row.status == "success"
    assert row.tpot_ms is None


@pytest.mark.parametrize(
    "frames,error",
    [
        ([(0, "data: {bad}\n\n")], "invalid_response"),
        ([(0, "data: [DONE]\n\n")], "incomplete_stream"),
        ([], "incomplete_stream"),
        ([(0, event({"choices": [{"text": "partial"}]}))], "incomplete_stream"),
        (final_frames(32, 0), "empty_response"),
        ([(0, event({"error": {"message": "bad request"}}))], "invalid_response"),
    ],
)
async def test_stream_failures(config, frames, error):
    row = await measure(config, frames)
    assert row.status == "failed"
    assert row.error_type == error


async def test_usage_missing_stays_missing(config):
    row = await measure(
        config, [(0, event({"choices": [{"text": "hello"}]}))] + final_frames(32, usage=False)
    )
    assert row.status == "success"
    assert row.output_tokens is None
    assert row.tpot_ms is None


@pytest.mark.parametrize("count,output", [(33, 4), (32, -1), (32, 100)])
async def test_inconsistent_usage_is_invalid(config, count, output):
    row = await measure(
        config, [(0, event({"choices": [{"text": "hello"}]}))] + final_frames(count, output)
    )
    assert row.error_type == "invalid_response"


async def test_request_timeout_preserves_record(config):
    config.servers[0].request_timeout_seconds = 0.01
    row = await measure(config, [(0.1, "data: [DONE]\n\n")])
    assert row.status == "timeout"
    assert row.completed_at_ms is not None


async def test_rate_limit(config):
    row = await measure(config, [(0, "rate limited")], status=429)
    assert row.error_type == "rate_limit"


async def test_error_finish_reason_cannot_succeed(config):
    row = await measure(
        config,
        [
            (0, event({"choices": [{"text": "partial", "finish_reason": "error"}]})),
            (0, "data: [DONE]\n\n"),
        ],
    )
    assert row.status == "failed"
    assert row.error_type == "server_finish_error"
