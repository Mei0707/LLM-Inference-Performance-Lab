"""Comparison and three-state regression policies; missing data never passes."""

import json
import math
import statistics
from pathlib import Path

import yaml
from pydantic import Field, model_validator

from inference_lab.config import StrictModel

COMPARISON_METRICS = [
    "requests_per_second",
    "output_tokens_per_second",
    "p95_ttft_ms",
    "p95_latency_ms",
    "error_rate",
]
REQUIRED_PROVENANCE = {
    "model_id",
    "model_revision",
    "tokenizer_id",
    "tokenizer_revision",
    "workload",
    "workload_sha256",
    "hardware",
    "simulation",
    "benchmark_source_sha256",
    "client_python",
    "client_libraries",
    "measurement_policy",
}


def compare(baseline, candidate):
    reasons = []
    for label, summary in [("baseline", baseline), ("candidate", candidate)]:
        if not summary.get("complete"):
            reasons.append(f"{label} run is incomplete")
        if not summary.get("comparability"):
            reasons.append(f"{label} lacks provenance")
        elif not summary["comparability"].get("hardware"):
            reasons.append(f"{label} lacks hardware identity")
        if summary.get("comparability"):
            missing = sorted(
                key
                for key in REQUIRED_PROVENANCE
                if key not in summary["comparability"] or summary["comparability"][key] is None
            )
            if missing:
                reasons.append(f"{label} lacks provenance fields: {', '.join(missing)}")
    left, right = baseline.get("comparability", {}), candidate.get("comparability", {})
    for key in sorted(set(left) | set(right)):
        if left.get(key) != right.get(key):
            reasons.append(f"Incompatible {key}")
    metrics = {}
    for metric in COMPARISON_METRICS:
        a, b = baseline.get(metric), candidate.get(metric)
        valid = all(isinstance(v, (int, float)) and math.isfinite(v) for v in [a, b])
        metrics[metric] = {
            "baseline": a,
            "candidate": b,
            "ratio": b / a if not reasons and valid and a != 0 else None,
        }
    resources = {}
    for metric in [
        "gpu_utilization_percent_mean",
        "gpu_memory_used_bytes_peak",
        "kv_cache_usage_fraction_mean",
    ]:
        resources[metric] = {
            "baseline": baseline.get("telemetry", {}).get(metric),
            "candidate": candidate.get("telemetry", {}).get(metric),
        }
    return {
        "comparable": not reasons,
        "reasons": reasons,
        "metrics": metrics,
        "resources": resources,
        "missing_metrics": [
            k for k, v in metrics.items() if v["baseline"] is None or v["candidate"] is None
        ],
    }


class Rule(StrictModel):
    metric: str
    minimum_ratio: float | None = Field(default=None, ge=0)
    maximum_ratio: float | None = Field(default=None, ge=0)
    maximum: float | None = None
    minimum: float | None = None

    @model_validator(mode="after")
    def has_threshold(self):
        if all(
            getattr(self, k) is None
            for k in ["minimum_ratio", "maximum_ratio", "maximum", "minimum"]
        ):
            raise ValueError("A regression rule requires at least one threshold")
        return self


class Policy(StrictModel):
    rules: list[Rule] = Field(min_length=1)
    baseline: str | None = None


def load_policy(path):
    data = yaml.safe_load(Path(path).read_text())
    return Policy.model_validate(data.get("regression", data))


def regression(baseline, candidate, policy):
    comparison = compare(baseline, candidate)
    checks = []
    for rule in policy.rules:
        a, b = baseline.get(rule.metric), candidate.get(rule.metric)
        status, reasons = "PASS", []
        if not comparison["comparable"]:
            status, reasons = "INCONCLUSIVE", comparison["reasons"]
        elif b is None or not isinstance(b, (int, float)) or not math.isfinite(b):
            status, reasons = "INCONCLUSIVE", ["Candidate metric is missing or nonfinite"]
        else:
            for bound in ["minimum_ratio", "maximum_ratio", "maximum", "minimum"]:
                threshold = getattr(rule, bound)
                if threshold is None:
                    continue
                value = b
                if bound.endswith("ratio"):
                    if (
                        a is None
                        or not isinstance(a, (int, float))
                        or not math.isfinite(a)
                        or a <= 0
                    ):
                        status, reasons = (
                            "INCONCLUSIVE",
                            ["Baseline ratio denominator is missing or nonpositive"],
                        )
                        break
                    value = b / a
                if (bound.startswith("minimum") and value < threshold) or (
                    bound.startswith("maximum") and value > threshold
                ):
                    status = "FAIL"
                    reasons.append(f"{bound}: observed {value:g}, threshold {threshold:g}")
        checks.append({"metric": rule.metric, "status": status, "reasons": reasons})
    status = (
        "INCONCLUSIVE"
        if any(c["status"] == "INCONCLUSIVE" for c in checks)
        else "FAIL"
        if any(c["status"] == "FAIL" for c in checks)
        else "PASS"
    )
    return {"status": status, "checks": checks, "comparison": comparison}


def repetition_summary(manifests_and_summaries):
    groups = {}
    for manifest, summary in manifests_and_summaries:
        key = json.dumps(
            {
                "server": manifest["server"],
                "workload": manifest["workload"],
                "model": manifest["model"],
                "provenance": summary.get("comparability"),
            },
            sort_keys=True,
        )
        groups.setdefault(key, []).append((manifest, summary))
    output = []
    for pairs in groups.values():
        manifest = pairs[0][0]
        group = {
            "server": manifest["server"]["name"],
            "workload": manifest["workload"],
            "model": manifest["model"],
            "run_ids": [s.get("run_id") for _, s in pairs],
            "repetitions": len(pairs),
            "complete_repetitions": sum(s["complete"] for _, s in pairs),
            "failed_requests": sum(s["failed_requests"] for _, s in pairs),
            "metrics": {},
        }
        for metric in COMPARISON_METRICS + [
            "slo_goodput_requests_per_second",
            "slo_attainment",
            "p95_schedule_latency_ms",
            "p50_tpot_ms",
        ]:
            values = [s[metric] for _, s in pairs if s["complete"] and s.get(metric) is not None]
            group["metrics"][metric] = {
                "median": statistics.median(values) if values else None,
                "min": min(values, default=None),
                "max": max(values, default=None),
                "repetitions_with_data": len(values),
            }
        output.append(group)
    return output
