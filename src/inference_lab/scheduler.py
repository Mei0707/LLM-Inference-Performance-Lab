"""Bounded concurrency, independent arrivals, and explicit overload/cancellation records."""

import asyncio
import random
import time

from inference_lab.client import elapsed_ms, finish_result, request
from inference_lab.metrics import RequestResult


def prepare_results(prompts, workload, seed, run_id):
    rng = random.Random(seed)
    scheduled = 0.0
    results = []
    for i, prompt in enumerate(prompts):
        if workload.mode == "open_loop" and i:
            scheduled += (
                1 / workload.arrival_rate
                if workload.arrival_distribution == "constant"
                else rng.expovariate(workload.arrival_rate)
            ) * 1000
        results.append(
            RequestResult(
                prompt.request_id,
                run_id,
                prompt.group,
                scheduled,
                prompt.input_tokens,
                prompt.max_output_tokens,
            )
        )
    return results


async def execute(client, server, model, workload, prompts, results, origin, on_result):
    queue = asyncio.Queue(maxsize=workload.max_pending)
    recorded = set()

    def save(result):
        if result.request_id not in recorded:
            on_result(result.to_dict())
            recorded.add(result.request_id)

    async def worker():
        while True:
            index = await queue.get()
            result = results[index]
            if workload.mode == "closed_loop":
                result.scheduled_at_ms = elapsed_ms(origin)
            try:
                await request(client, server, model, workload, prompts[index], result, origin)
            finally:
                queue.task_done()
                save(result)

    try:
        async with asyncio.TaskGroup() as group:
            workers = [group.create_task(worker()) for _ in range(workload.concurrency)]
            for i, result in enumerate(results):
                if workload.mode == "open_loop":
                    delay = origin + result.scheduled_at_ms / 1000 - time.perf_counter()
                    if delay > 0:
                        await asyncio.sleep(delay)
                    try:
                        queue.put_nowait(i)
                    except asyncio.QueueFull:
                        result.status, result.error_type = "dropped", "client_backlog_full"
                        finish_result(result, origin)
                        save(result)
                    # Allow workers to dispatch even when the producer is behind schedule.
                    await asyncio.sleep(0)
                else:
                    await queue.put(i)
            await queue.join()
            for task in workers:
                task.cancel()
    finally:
        for result in results:
            if result.status == "pending":
                result.status, result.error_type = "canceled", "not_dispatched"
                # Future scheduled arrivals have not happened yet.
                result.completed_at_ms = elapsed_ms(origin)
            save(result)
