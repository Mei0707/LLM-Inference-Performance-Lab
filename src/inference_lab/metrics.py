"""Raw request observations and aggregation with explicit denominators."""

import math
from collections import Counter
from dataclasses import asdict, dataclass


@dataclass
class RequestResult:
    request_id: str
    experiment_id: str
    group: str
    scheduled_at_ms: float
    input_tokens: int
    requested_output_tokens: int
    dispatched_at_ms: float | None = None
    completed_at_ms: float | None = None
    output_tokens: int | None = None
    server_input_tokens: int | None = None
    ttft_ms: float | None = None
    tpot_ms: float | None = None
    latency_ms: float | None = None
    schedule_latency_ms: float | None = None
    dispatch_delay_ms: float | None = None
    first_content_at_ms: float | None = None
    last_content_at_ms: float | None = None
    content_events: int = 0
    status: str = "pending"
    error_type: str | None = None
    error_message: str | None = None
    http_status: int | None = None
    finish_reason: str | None = None
    usage_source: str | None = None
    output_text: str = ""

    def to_dict(self):
        return asdict(self)


def percentile(values, p):
    values = sorted(v for v in values if v is not None and math.isfinite(v))
    if not values:
        return None
    position = (len(values) - 1) * p / 100
    low, high = math.floor(position), math.ceil(position)
    return values[low] + (values[high] - values[low]) * (position - low)


def summarize(rows, duration_seconds, report):
    success = [r for r in rows if r["status"] == "success"]
    known = [r for r in success if r["output_tokens"] is not None]
    denominator = duration_seconds if duration_seconds > 0 else None
    totals = sum(r["output_tokens"] for r in known)
    result = {
        "schema_version": 1,
        "requests": len(rows),
        "attempted_requests": sum(r["dispatched_at_ms"] is not None for r in rows),
        "successful_requests": len(success),
        "failed_requests": len(rows) - len(success),
        "status_counts": dict(Counter(r["status"] for r in rows)),
        "error_counts": dict(Counter(r["error_type"] for r in rows if r["error_type"])),
        "error_rate": (len(rows) - len(success)) / len(rows) if rows else None,
        "duration_seconds": duration_seconds,
        "requests_per_second": len(success) / denominator if denominator else None,
        "successful_output_tokens_known": totals,
        "token_usage_coverage": len(known) / len(success) if success else None,
        "output_tokens_per_second": totals / denominator
        if denominator and len(known) == len(success) and success
        else None,
        "input_tokens_per_second": sum(r["input_tokens"] for r in success) / denominator
        if denominator
        else None,
        "empty_responses": sum(r["error_type"] == "empty_response" for r in rows),
        "truncated_responses": sum(r["finish_reason"] == "length" for r in rows),
        "notes": [
            "TPOT is a client chunk-boundary estimate using server token counts; chunks may contain multiple tokens."
        ],
    }
    for metric in ["ttft_ms", "tpot_ms", "latency_ms", "schedule_latency_ms", "dispatch_delay_ms"]:
        observed = [r[metric] for r in success if r[metric] is not None]
        result[f"{metric}_samples"] = len(observed)
        for p in sorted(set([50, 95, 99] + report.percentiles)):
            value = percentile(observed, p)
            if p >= 99 and len(observed) < report.p99_min_samples:
                value = None
            result[f"p{p:g}_{metric}"] = value
    slo_enabled = report.slo_ttft_ms is not None or report.slo_latency_ms is not None
    good = [
        r
        for r in success
        if (
            report.slo_ttft_ms is None
            or (r["ttft_ms"] is not None and r["ttft_ms"] <= report.slo_ttft_ms)
        )
        and (report.slo_latency_ms is None or r["schedule_latency_ms"] <= report.slo_latency_ms)
    ]
    result["slo_goodput_requests_per_second"] = (
        len(good) / denominator if slo_enabled and denominator else None
    )
    result["slo_attainment"] = len(good) / len(rows) if slo_enabled and rows else None
    result["total_tokens_per_second"] = (
        result["input_tokens_per_second"] + result["output_tokens_per_second"]
        if result["input_tokens_per_second"] is not None
        and result["output_tokens_per_second"] is not None
        else None
    )
    result["groups"] = {}
    for group in sorted({r["group"] for r in rows}):
        group_rows = [r for r in rows if r["group"] == group]
        ok = [r for r in group_rows if r["status"] == "success"]
        result["groups"][group] = {
            "requests": len(group_rows),
            "successes": len(ok),
            "p95_latency_ms": percentile([r["latency_ms"] for r in ok], 95),
        }
    return result
