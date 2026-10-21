"""Regenerable Markdown and figures, labeled with provenance and missing observations."""

import json
import os
import statistics
from pathlib import Path

from inference_lab.analysis import COMPARISON_METRICS, repetition_summary
from inference_lab.storage import read_parquet, write_json
from inference_lab.telemetry import summarize_telemetry


def number(value):
    return (
        "unavailable"
        if value is None
        else f"{value:.3f}"
        if isinstance(value, float)
        else str(value)
    )


def plotter():
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/inference-lab-matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def label(manifest, varying=None):
    gpu = manifest.get("observed_gpu", [])
    hardware = (
        ", ".join(g["name"] for g in gpu)
        if gpu
        else manifest["server"]["runtime"].get("gpu_model") or "GPU unobserved"
    )
    if manifest["simulation"]:
        hardware = "MOCK SERVER / CPU protocol test"
    w = manifest["workload"]
    dimensions = {
        "input_tokens": f"input={w['input_tokens']}",
        "max_output_tokens": f"output limit={w['max_output_tokens']}",
        "concurrency": f"concurrency={w['concurrency']}",
    }
    controls = ", ".join(value for key, value in dimensions.items() if key != varying)
    return f"{manifest['model']['id']} | {hardware}\n{w['mode']}, {controls}"


def run_report(directory, manifest, summary, rows):
    directory = Path(directory)
    lines = [
        f"# {manifest['experiment_name']} / {manifest['run_id']}",
        "",
        "**SIMULATION: these measurements validate the client and are not model or GPU performance results.**"
        if manifest["simulation"]
        else "Real-server benchmark; review runtime metadata completeness before drawing conclusions.",
        "",
        f"State: **{manifest['state']}**. {manifest.get('failure_reason') or ''}",
        "",
        f"Model: `{manifest['model']['id']}` at `{manifest['model']['revision']}`.",
        "",
        f"Hardware and workload: {label(manifest).replace(chr(10), '; ')}.",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
    ]
    for key in [
        "requests",
        "attempted_requests",
        "successful_requests",
        "failed_requests",
        "error_rate",
        "duration_seconds",
        "requests_per_second",
        "output_tokens_per_second",
        "token_usage_coverage",
        "p50_ttft_ms",
        "p95_ttft_ms",
        "p99_ttft_ms",
        "p50_tpot_ms",
        "p95_latency_ms",
        "p95_schedule_latency_ms",
        "p95_dispatch_delay_ms",
        "slo_goodput_requests_per_second",
    ]:
        lines.append(f"| {key} | {number(summary.get(key))} |")
    lines += [
        "",
        "Throughput uses the entire measured workload interval, including arrival gaps and drain. Warm-up and telemetry shutdown are excluded.",
        "",
        "TPOT uses first/last content-event timestamps and final server token usage. It is an estimate; a single content chunk cannot identify token spacing. Missing usage leaves token throughput unavailable. p99 requires the configured minimum sample count.",
        "",
        f"Status counts: `{json.dumps(summary['status_counts'], sort_keys=True)}`.",
        "",
        f"Error counts: `{json.dumps(summary['error_counts'], sort_keys=True)}`.",
        "",
        f"Telemetry: `{json.dumps(summary['telemetry'], sort_keys=True)}`.",
        "",
        "See manifest.json for declared versus observed runtime versions and raw workload provenance.",
        "",
    ]
    latencies = [
        r["latency_ms"] for r in rows if r["status"] == "success" and r["latency_ms"] is not None
    ]
    if latencies:
        plt = plotter()
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.hist(latencies, bins=min(30, max(1, len(latencies))))
        ax.set(
            xlabel="Client service latency (ms)",
            ylabel="Successful requests",
            title=label(manifest),
        )
        fig.tight_layout()
        fig.savefig(directory / "latency.png", dpi=140)
        plt.close(fig)
        lines += ["![Service latency distribution](latency.png)", ""]
    (directory / "report.md").write_text("\n".join(lines))


def experiment_report(destination, directories):
    destination = Path(destination)
    pairs = [
        (
            json.loads((d / "manifest.json").read_text()),
            json.loads((d / "summary.json").read_text()),
        )
        for d in directories
    ]
    groups = repetition_summary(pairs)
    write_json(destination / "aggregate.json", {"groups": groups})
    lines = [
        "# Experiment comparison",
        "",
        "Medians and ranges are calculated across completed runs, not by pooling requests.",
        "",
    ]
    if any(m["simulation"] for m, _ in pairs):
        lines += ["**SIMULATION: mock protocol measurements; no GPU performance claims.**", ""]
    lines += [
        "| Server | Concurrency | Input tokens | Runs complete/total | Requests/s median [min, max] | Output tokens/s | p95 TTFT (ms) | Failed requests |",
        "| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |",
    ]
    for group in groups:
        metrics = group["metrics"]
        rps = metrics["requests_per_second"]
        lines.append(
            f"| {group['server']} | {group['workload']['concurrency']} | {group['workload']['input_tokens']} | {group['complete_repetitions']}/{group['repetitions']} | {number(rps['median'])} [{number(rps['min'])}, {number(rps['max'])}] | {number(metrics['output_tokens_per_second']['median'])} | {number(metrics['p95_ttft_ms']['median'])} | {group['failed_requests']} |"
        )
    lines += ["", "Individual runs:", ""]
    lines += [
        f"- [{m['run_id']}]({d.name}/report.md) — {m['state']}"
        for d, (m, _) in zip(directories, pairs, strict=True)
    ]
    goodput = [
        g for g in groups if g["metrics"]["slo_goodput_requests_per_second"]["median"] is not None
    ]
    if goodput:
        best = max(goodput, key=lambda g: g["metrics"]["slo_goodput_requests_per_second"]["median"])
        lines += [
            "",
            f"Highest observed median SLO goodput: **{best['server']}**, concurrency={best['workload']['concurrency']}, input={best['workload']['input_tokens']}: {number(best['metrics']['slo_goodput_requests_per_second']['median'])} requests/s. This selects among the measured configurations; it does not establish a long-term sustainable limit.",
        ]
    # Group curves by every controlled dimension except the plotted independent variable.
    # This prevents connecting different output lengths or servers into a misleading line.
    plt = plotter()
    charts = [
        ("concurrency", "requests_per_second", "Requests/s"),
        ("concurrency", "output_tokens_per_second", "Output tokens/s"),
        ("concurrency", "p95_ttft_ms", "p95 TTFT (ms)"),
        ("input_tokens", "p95_ttft_ms", "p95 TTFT (ms)"),
    ]
    for dimension, metric, ylabel in charts:
        series = {}
        for m, s in pairs:
            if not s["complete"] or s.get(metric) is None:
                continue
            controls = {k: v for k, v in m["workload"].items() if k != dimension}
            key = json.dumps(
                {
                    "controls": controls,
                    "server": m["server"],
                    "model": m["model"],
                    "hardware": s.get("comparability", {}).get("hardware"),
                    "source": s.get("comparability", {}).get("benchmark_source_sha256"),
                },
                sort_keys=True,
            )
            series.setdefault(key, []).append((m["workload"][dimension], s[metric], m))
        useful = [points for points in series.values() if len({p[0] for p in points}) > 1]
        if not useful:
            continue
        fig, ax = plt.subplots(figsize=(10, 6))
        for points in useful:
            xs = sorted({p[0] for p in points})
            ys = [statistics.median(p[1] for p in points if p[0] == x) for x in xs]
            m = points[0][2]
            ax.plot(
                xs,
                ys,
                marker="o",
                label=f"{m['server']['name']} / in={m['workload']['input_tokens']} / out={m['workload']['max_output_tokens']}",
            )
        ax.set(
            xlabel=f"{dimension} ({'tokens' if dimension == 'input_tokens' else 'requests'})",
            ylabel=ylabel,
            title=label(useful[0][0][2], varying=dimension)
            + "\nMedians across repetitions; see table for ranges",
        )
        ax.legend(fontsize=8)
        fig.tight_layout()
        name = f"{dimension}-vs-{metric}.png"
        fig.savefig(destination / name, dpi=140)
        plt.close(fig)
        lines += ["", f"![{dimension} vs {ylabel}]({name})"]
    # Configuration/precision comparison bars retain the workload on each tick.
    if len({m["server"]["name"] for m, _ in pairs}) > 1:
        visible = [
            g for g in groups if g["metrics"]["output_tokens_per_second"]["median"] is not None
        ]
        if visible:
            fig, ax = plt.subplots(figsize=(max(10, len(visible) * 1.3), 6))
            names = [
                f"{g['server']}\n{g['model']['dtype']} / {g['model']['quantization'] or 'unquantized'}\nin={g['workload']['input_tokens']}, c={g['workload']['concurrency']}"
                for g in visible
            ]
            ax.bar(names, [g["metrics"]["output_tokens_per_second"]["median"] for g in visible])
            ax.set(
                ylabel="Median output tokens/s",
                title=label(pairs[0][0])
                + "\nConfiguration observations; inspect manifests for controlled variables",
            )
            ax.tick_params(axis="x", labelsize=8)
            fig.tight_layout()
            fig.savefig(destination / "configuration-throughput.png", dpi=140)
            plt.close(fig)
            lines += [
                "",
                "![Batching / precision configuration throughput](configuration-throughput.png)",
            ]
        memory_groups = {}
        for m, s in pairs:
            memory = s.get("telemetry", {}).get("gpu_memory_used_bytes_mean")
            if s["complete"] and memory is not None:
                w = m["workload"]
                name = f"{m['server']['name']} / {m['model']['quantization'] or m['model']['dtype']}\nin={w['input_tokens']}, out={w['max_output_tokens']}, c={w['concurrency']}"
                memory_groups.setdefault(name, []).append(memory / 1024**3)
        if memory_groups:
            fig, ax = plt.subplots(figsize=(max(10, len(memory_groups) * 1.3), 6))
            ax.bar(
                list(memory_groups),
                [statistics.median(values) for values in memory_groups.values()],
            )
            ax.set(
                ylabel="Median of per-run mean GPU memory (GiB)",
                title=label(pairs[0][0]) + "\nConfiguration / precision memory observations",
            )
            ax.tick_params(axis="x", labelsize=8)
            fig.tight_layout()
            fig.savefig(destination / "configuration-memory.png", dpi=140)
            plt.close(fig)
            lines += ["", "![Configuration / precision GPU memory](configuration-memory.png)"]
    telemetry_charts = [
        (
            "gpu_memory_used_bytes_mean",
            "concurrency",
            "GPU memory mean (GiB)",
            "Concurrent worker limit (requests)",
            "gpu-memory-vs-concurrency.png",
        ),
        (
            "kv_cache_usage_fraction_mean",
            "p95_latency_ms",
            "Mean KV-cache occupancy (fraction)",
            "p95 service latency (ms)",
            "kv-cache-vs-tail-latency.png",
        ),
    ]
    for resource, other, resource_label, other_label, filename in telemetry_charts:
        points = [
            (m, s)
            for m, s in pairs
            if s["complete"]
            and s.get("telemetry", {}).get(resource) is not None
            and (other == "concurrency" or s.get(other) is not None)
        ]
        if not points:
            continue
        fig, ax = plt.subplots(figsize=(10, 6))
        for m, s in points:
            value = s["telemetry"][resource]
            resource_value = value / 1024**3 if resource.startswith("gpu_memory") else value
            x, y = (
                (m["workload"]["concurrency"], resource_value)
                if other == "concurrency"
                else (resource_value, s[other])
            )
            ax.scatter(
                x,
                y,
                label=f"{m['server']['name']} / {m['model']['quantization'] or m['model']['dtype']} / in={m['workload']['input_tokens']}",
            )
        ax.set(
            xlabel=other_label if other == "concurrency" else resource_label,
            ylabel=resource_label if other == "concurrency" else other_label,
            title=label(points[0][0], varying="concurrency")
            + "\nPer-run observations; correlation does not establish a bottleneck",
        )
        handles, names = ax.get_legend_handles_labels()
        unique = dict(zip(names, handles, strict=True))
        ax.legend(unique.values(), unique.keys(), fontsize=8)
        fig.tight_layout()
        fig.savefig(destination / filename, dpi=140)
        plt.close(fig)
        lines += ["", f"![{resource_label}]({filename})"]
    (destination / "report.md").write_text("\n".join(lines) + "\n")


def regenerate(directory):
    directory = Path(directory)
    if (directory / "manifest.json").exists():
        manifest = json.loads((directory / "manifest.json").read_text())
        rows = read_parquet(directory / "requests.parquet")
        from inference_lab.config import Report
        from inference_lab.metrics import summarize

        summary_path = directory / "summary.json"
        previous = json.loads(summary_path.read_text()) if summary_path.exists() else {}
        duration = manifest.get("measurement_duration_seconds", previous.get("duration_seconds"))
        if duration is None:
            raise ValueError("Run manifest lacks a measurement duration")
        summary = summarize(rows, duration, Report.model_validate(manifest["report_config"]))
        from inference_lab.controller import comparability

        summary.update(
            {
                "run_id": manifest["run_id"],
                "complete": manifest["state"] == "completed",
                "simulation": manifest["simulation"],
                "comparability": comparability(manifest),
                "report_config": manifest["report_config"],
                "telemetry": summarize_telemetry(
                    read_parquet(directory / "gpu_metrics.parquet"),
                    read_parquet(directory / "server_metrics.parquet"),
                    manifest.get("telemetry_errors", []),
                ),
            }
        )
        write_json(directory / "summary.json", summary)
        run_report(directory, manifest, summary, rows)
    else:
        directories = sorted(p.parent for p in directory.glob("*/manifest.json"))
        if not directories:
            raise ValueError("No benchmark artifacts found")
        for run in directories:
            regenerate(run)
        experiment_report(directory, directories)


def comparison_markdown(result):
    lines = ["# Benchmark comparison", "", f"Comparable: **{result['comparable']}**.", ""]
    lines += [f"- {reason}" for reason in result["reasons"]]
    lines += [
        "",
        "| Metric | Baseline | Candidate | Candidate / baseline |",
        "| --- | ---: | ---: | ---: |",
    ]
    for key in COMPARISON_METRICS:
        row = result["metrics"][key]
        lines.append(
            f"| {key} | {number(row['baseline'])} | {number(row['candidate'])} | {number(row['ratio'])} |"
        )
    lines += [
        "",
        f"Resource observations: `{json.dumps(result['resources'], sort_keys=True)}`.",
        "",
        "Ratios are suppressed for incompatible or incomplete runs. Single-run differences require confirmation across repetitions.",
    ]
    return "\n".join(lines) + "\n"
