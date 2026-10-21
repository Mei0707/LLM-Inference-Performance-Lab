import asyncio
import time

import pytest

from inference_lab.scheduler import execute, prepare_results
from inference_lab.workload import MockTokenizer, generate


async def simulate(config, monkeypatch, delay=0.015, cancel_after=None):
    workload = config.workload
    prompts = generate(workload, MockTokenizer(), 42)
    rows = prepare_results(prompts, workload, 42, "run")
    active = peak = 0
    started = []
    saved = []

    async def fake_request(client, server, model, workload, prompt, row, origin):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        row.dispatched_at_ms = (time.perf_counter() - origin) * 1000
        started.append(row.request_id)
        try:
            await asyncio.sleep(delay)
            row.status = "success"
        except asyncio.CancelledError:
            row.status, row.error_type = "canceled", "canceled"
            raise
        finally:
            active -= 1

    monkeypatch.setattr("inference_lab.scheduler.request", fake_request)
    task = asyncio.create_task(
        execute(
            None,
            config.servers[0],
            config.model,
            workload,
            prompts,
            rows,
            time.perf_counter(),
            saved.append,
        )
    )
    if cancel_after:
        await asyncio.sleep(cancel_after)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    else:
        await task
    return rows, saved, peak, started


async def test_concurrency_and_count(config, monkeypatch):
    config.workload.requests = 15
    rows, saved, peak, started = await simulate(config, monkeypatch)
    assert peak == config.workload.concurrency
    assert len(saved) == len(started) == len(rows) == 15
    assert all(row.status == "success" for row in rows)


async def test_open_loop_independent_arrivals_and_client_delay(config, monkeypatch):
    config.workload.mode = "open_loop"
    config.workload.arrival_rate = 1000
    config.workload.concurrency = 1
    rows, _, peak, _ = await simulate(config, monkeypatch)
    assert [r.scheduled_at_ms for r in rows] == [0, 1, 2, 3, 4, 5]
    assert rows[-1].dispatched_at_ms - rows[-1].scheduled_at_ms > 20
    assert peak == 1


async def test_backlog_drops_are_visible(config, monkeypatch):
    config.workload.mode = "open_loop"
    config.workload.arrival_rate = 10000
    config.workload.max_pending = 1
    config.workload.concurrency = 1
    rows, saved, _, _ = await simulate(config, monkeypatch)
    assert any(r.status == "dropped" for r in rows)
    assert len(saved) == len(rows)


async def test_cancel_preserves_all_requests(config, monkeypatch):
    config.workload.requests = 30
    rows, saved, _, started = await simulate(config, monkeypatch, delay=1, cancel_after=0.01)
    assert len(rows) == len(saved) == 30
    assert len({r["request_id"] for r in saved}) == 30
    assert len(started) == 2
    assert all(r.status == "canceled" for r in rows)
    assert any(r.error_type == "not_dispatched" for r in rows)


def test_poisson_seed(config):
    config.workload.mode = "open_loop"
    config.workload.arrival_rate = 3
    config.workload.arrival_distribution = "poisson"
    prompts = generate(config.workload, MockTokenizer(), 42)
    a = prepare_results(prompts, config.workload, 42, "a")
    b = prepare_results(prompts, config.workload, 42, "b")
    assert [r.scheduled_at_ms for r in a] == [r.scheduled_at_ms for r in b]
    assert a[-1].scheduled_at_ms > a[0].scheduled_at_ms
