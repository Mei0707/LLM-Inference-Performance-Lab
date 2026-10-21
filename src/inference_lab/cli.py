"""Command-line entry point; importing the package has no network or GPU side effects."""

import argparse
import asyncio
import json
import shlex
import socket
from pathlib import Path
from urllib.parse import urlsplit

from inference_lab.analysis import compare, load_policy, regression
from inference_lab.config import Config, load_config
from inference_lab.controller import run_experiment
from inference_lab.reporting import comparison_markdown, regenerate
from inference_lab.storage import write_json


def demo_config(url):
    return Config.model_validate(
        {
            "experiment": {"name": "mock-concurrency", "repetitions": 3, "warmup_requests": 2},
            "model": {"id": "mock-model", "revision": "mock-v1", "tokenizer": "mock"},
            "servers": [{"base_url": url, "readiness_timeout_seconds": 5}],
            "workload": {
                "requests": 24,
                "input_tokens": 32,
                "max_output_tokens": 8,
                "concurrency": [1, 4, 8],
            },
            "metrics": {"collect_gpu": False, "sample_interval_ms": 50},
        }
    )


async def demo(output):
    import uvicorn

    from inference_lab.mock_server import create_mock_app

    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(create_mock_app(), log_level="warning"))
    task = asyncio.create_task(server.serve(sockets=[sock]))
    try:
        while not server.started:
            if task.done():
                await task
                raise RuntimeError("Mock server did not start")
            await asyncio.sleep(0.01)
        return await run_experiment(demo_config(f"http://127.0.0.1:{port}"), output)
    finally:
        server.should_exit = True
        await task
        sock.close()


def serve_command(config, name):
    server = next((s for s in config.servers if s.name == name), None)
    if server is None:
        raise ValueError(f"Unknown server variant {name!r}")
    model = server.model or config.model
    if model.tokenizer == "mock":
        return "inference-lab mock-server"
    args = [
        "vllm",
        "serve",
        model.id,
        "--revision",
        model.revision,
        "--tokenizer",
        model.tokenizer_id or model.id,
        "--tokenizer-revision",
        model.tokenizer_revision or model.revision,
        "--dtype",
        model.dtype,
        "--max-model-len",
        str(model.max_model_len),
        "--generation-config",
        "vllm",
    ]
    if model.quantization:
        args += ["--quantization", model.quantization]
    if "port" not in server.serving_flags and urlsplit(server.base_url).port is not None:
        args += ["--port", str(urlsplit(server.base_url).port)]
    reserved = {
        "model",
        "revision",
        "tokenizer",
        "tokenizer-revision",
        "dtype",
        "max-model-len",
        "quantization",
        "generation-config",
    }
    for key, value in server.serving_flags.items():
        key = key.replace("_", "-")
        if key in reserved or key.startswith("-"):
            raise ValueError(f"Flag {key!r} conflicts with model metadata")
        args += [f"--{key}"]
        if not isinstance(value, bool):
            args += [str(value)]
        elif not value:
            args[-1] = f"--no-{key}"
    return shlex.join(args)


def parser():
    root = argparse.ArgumentParser(description="Reproducible LLM inference benchmark lab")
    commands = root.add_subparsers(dest="command", required=True)
    for name in ["validate", "run", "serve-command"]:
        sub = commands.add_parser(name)
        sub.add_argument("config", type=Path)
        if name == "run":
            sub.add_argument("--output", type=Path, default=Path("results"))
            sub.add_argument("--server", help="Run one already configured server variant")
        if name == "serve-command":
            sub.add_argument("--server", default="baseline")
    sub = commands.add_parser("demo", help="Run a local MOCK streaming benchmark and reports")
    sub.add_argument("--output", type=Path, default=Path("results"))
    sub = commands.add_parser("mock-server")
    sub.add_argument("--host", default="127.0.0.1")
    sub.add_argument("--port", type=int, default=8000)
    sub = commands.add_parser("report", help="Regenerate reports from raw artifacts")
    sub.add_argument("directory", type=Path)
    for name in ["compare", "check"]:
        sub = commands.add_parser(name)
        sub.add_argument("baseline", type=Path, help="Baseline summary.json")
        sub.add_argument("candidate", type=Path, help="Candidate summary.json")
        sub.add_argument("--output", type=Path)
        if name == "check":
            sub.add_argument("--policy", type=Path, required=True)
    return root


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command in {"validate", "run", "serve-command"}:
            config = load_config(args.config)
            if args.command == "validate":
                print(f"Valid: {sum(1 for _ in config.runs())} runs")
            elif args.command == "serve-command":
                print(serve_command(config, args.server))
            else:
                if args.server:
                    config.servers = [s for s in config.servers if s.name == args.server]
                    if not config.servers:
                        raise ValueError(f"Unknown server variant {args.server!r}")
                directory = asyncio.run(run_experiment(config, args.output))
                print(directory / "report.md")
                summaries = [json.loads(p.read_text()) for p in directory.glob("*/summary.json")]
                return (
                    0
                    if summaries
                    and all(s["complete"] and s["failed_requests"] == 0 for s in summaries)
                    else 1
                )
        elif args.command == "demo":
            directory = asyncio.run(demo(args.output))
            print(directory / "report.md")
            summaries = [json.loads(p.read_text()) for p in directory.glob("*/summary.json")]
            return (
                0
                if summaries and all(s["complete"] and s["failed_requests"] == 0 for s in summaries)
                else 1
            )
        elif args.command == "mock-server":
            import uvicorn

            uvicorn.run("inference_lab.mock_server:app", host=args.host, port=args.port)
        elif args.command == "report":
            regenerate(args.directory)
            print(args.directory / "report.md")
        elif args.command in {"compare", "check"}:
            a, b = (json.loads(p.read_text()) for p in [args.baseline, args.candidate])
            if args.command == "compare":
                result = compare(a, b)
                output = comparison_markdown(result)
                if args.output:
                    args.output.write_text(output)
                print(output)
                return 0 if result["comparable"] else 2
            result = regression(a, b, load_policy(args.policy))
            if args.output:
                write_json(args.output, result)
            print(json.dumps(result, indent=2))
            return {"PASS": 0, "FAIL": 1, "INCONCLUSIVE": 2}[result["status"]]
    except KeyboardInterrupt:
        print("Interrupted; finalized partial artifacts are preserved under the output directory.")
        return 130
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"Error: {exc}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
