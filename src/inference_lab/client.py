"""OpenAI completions SSE client. Content events and token usage are separate observations."""

import asyncio
import json
import time

import httpx

from inference_lab.metrics import RequestResult


class InvalidResponse(ValueError):
    pass


async def sse_data(response):
    """Parse SSE data fields across transport chunks, comments, and CRLF lines."""
    fields = []
    async for line in response.aiter_lines():
        if line == "":
            if fields:
                yield "\n".join(fields)
                fields = []
        elif line.startswith("data:"):
            fields.append(line[5:].removeprefix(" "))
    if fields:
        yield "\n".join(fields)


def elapsed_ms(origin):
    return (time.perf_counter() - origin) * 1000


def finish_result(result, origin):
    result.completed_at_ms = elapsed_ms(origin)
    result.schedule_latency_ms = result.completed_at_ms - result.scheduled_at_ms
    if result.dispatched_at_ms is not None:
        result.latency_ms = result.completed_at_ms - result.dispatched_at_ms
        result.dispatch_delay_ms = result.dispatched_at_ms - result.scheduled_at_ms
    if result.first_content_at_ms is not None:
        result.ttft_ms = result.first_content_at_ms - result.dispatched_at_ms
        if result.output_tokens and result.output_tokens >= 2 and result.content_events >= 2:
            result.tpot_ms = (result.last_content_at_ms - result.first_content_at_ms) / (
                result.output_tokens - 1
            )


async def request(client, server, model, workload, prompt, result: RequestResult, origin):
    result.dispatched_at_ms = elapsed_ms(origin)
    payload = {
        "model": model.id,
        "prompt": prompt.token_ids,
        "max_tokens": prompt.max_output_tokens,
        "temperature": workload.temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
        "seed": workload.sampling_seed,
    }
    if workload.ignore_eos:
        payload["ignore_eos"] = True
    try:
        async with asyncio.timeout(server.request_timeout_seconds):
            async with client.stream("POST", "/v1/completions", json=payload) as response:
                result.http_status = response.status_code
                if response.status_code != 200:
                    body = (await response.aread()).decode(errors="replace")[:1000]
                    result.status = "failed"
                    result.error_type = (
                        "rate_limit"
                        if response.status_code == 429
                        else "out_of_memory"
                        if "out of memory" in body.lower()
                        else "http_error"
                    )
                    result.error_message = f"HTTP {response.status_code}: {body}"
                    return
                done = False
                async for data in sse_data(response):
                    if data.strip() == "[DONE]":
                        done = True
                        break
                    event = json.loads(data)
                    if not isinstance(event, dict):
                        raise InvalidResponse("SSE event must be an object")
                    if event.get("error"):
                        raise InvalidResponse(str(event["error"])[:1000])
                    usage = event.get("usage")
                    if usage:
                        for name in ["prompt_tokens", "completion_tokens"]:
                            if type(usage.get(name)) is not int or usage[name] < 0:
                                raise InvalidResponse(f"Invalid usage.{name}")
                        result.output_tokens = usage["completion_tokens"]
                        result.server_input_tokens = usage["prompt_tokens"]
                        result.usage_source = "server"
                    choices = event.get("choices", [])
                    if not isinstance(choices, list) or len(choices) > 1:
                        raise InvalidResponse("Expected zero or one completion choice")
                    for choice in choices:
                        text = choice.get("text", choice.get("delta", {}).get("content", ""))
                        if text is not None and not isinstance(text, str):
                            raise InvalidResponse("Completion content must be text")
                        if text:
                            timestamp = elapsed_ms(origin)
                            if result.first_content_at_ms is None:
                                result.first_content_at_ms = timestamp
                            result.last_content_at_ms = timestamp
                            result.content_events += 1
                            result.output_text += text
                        if choice.get("finish_reason") is not None:
                            result.finish_reason = choice["finish_reason"]
                if not done or result.finish_reason is None:
                    result.status, result.error_type = "failed", "incomplete_stream"
                elif result.finish_reason not in {"stop", "length"}:
                    result.status, result.error_type = "failed", "server_finish_error"
                elif not result.output_text:
                    result.status, result.error_type = "failed", "empty_response"
                elif (
                    result.server_input_tokens is not None
                    and result.server_input_tokens != prompt.input_tokens
                ):
                    raise InvalidResponse(
                        "Server prompt token count differs from submitted token IDs"
                    )
                elif (
                    result.output_tokens is not None
                    and not 0 < result.output_tokens <= prompt.max_output_tokens
                ):
                    raise InvalidResponse("Server completion count outside requested output limit")
                else:
                    result.status = "success"
    except (TimeoutError, httpx.TimeoutException):
        result.status, result.error_type = "timeout", "timeout"
    except asyncio.CancelledError:
        result.status, result.error_type = "canceled", "canceled"
        raise
    except httpx.RemoteProtocolError as exc:
        result.status, result.error_type = "failed", "incomplete_stream"
        result.error_message = str(exc)[:1000]
    except httpx.HTTPError as exc:
        result.status, result.error_type = "failed", "connection_error"
        result.error_message = str(exc)[:1000]
    except (ValueError, TypeError, KeyError, AttributeError) as exc:
        result.status, result.error_type = "failed", "invalid_response"
        result.error_message = str(exc)[:1000]
    finally:
        finish_result(result, origin)
