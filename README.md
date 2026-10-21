# LLM Inference Performance Lab

A Python platform for measuring streaming inference, running repeatable workload matrices, collecting GPU and server telemetry, and comparing performance with explicit provenance.

The implementation includes a CPU-only mock streaming server. **Mock results are protocol tests, not LLM/GPU benchmark results.** Real NVIDIA measurements and an optimization case study require running the included GPU experiment workflow on compatible hardware. See [proposal review](docs/proposal-review.md) for the design decisions and acceptance status.

## Quick start (Python 3.11+)

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.lock
python -m pip install --no-build-isolation --no-deps -e .
inference-lab demo
```

The demo starts a temporary loopback HTTP server, runs concurrency 1/4/8 three times each, writes raw observations and reports under `results/mock-concurrency-…/`, then stops the server. Open the printed `report.md` and its linked PNG charts. No model weights, GPU, or API key are required. The lock captures the tested Python 3.11 client environment; GPU/tokenizer extras are installed separately.

```sh
inference-lab mock-server --port 8000
# In a second terminal:
inference-lab validate configs/mock.yaml
inference-lab run configs/mock.yaml
inference-lab report results/EXPERIMENT_DIRECTORY
```

## Real NVIDIA benchmarks

Use a Linux NVIDIA host with the NVIDIA Container Toolkit and Docker Compose GPU support. The included example pins Qwen2.5-1.5B-Instruct and vLLM 0.12.0 as a reproducible starting configuration, rather than a claim about the latest runtime. Validate fit and kernel compatibility on your GPU.

```sh
cp .env.example .env
sh scripts/verify_gpu.sh
docker compose --profile gpu up -d vllm dcgm prometheus grafana
python -m pip install -e '.[tokenizer,gpu]'
inference-lab validate configs/concurrency.yaml
inference-lab run configs/concurrency.yaml
```

NVML measures the machine running the client. Run the client on the GPU host (or use the `bench-gpu` Compose service, which has GPU access). A client on a Mac cannot collect a remote server's GPU metrics through NVML. DCGM supplies GPU dashboard metrics independently. Populate `servers[].runtime` in your experiment YAML from actual observations; record the resolved image digest using `docker image inspect`, and versions using `docker compose exec vllm python -c 'import torch,vllm; print(torch.__version__,torch.version.cuda,vllm.__version__)'`. Model revision and serving flags are declared metadata; the API does not prove that a live server loaded that revision. Verify the launch command and container inspect output before publishing a result.

```sh
inference-lab serve-command configs/batching.yaml --server small-batch
docker compose --profile gpu run --rm --use-aliases bench-gpu
```

`serve-command` prints the intended command. Server variants represent **separately configured, already running endpoints**. The benchmark never claims to change a live server's flags. For a single GPU, run variants sequentially, restarting vLLM between configurations; see [experiments](docs/experiments.md).

## Artifacts and comparisons

Each run writes `manifest.json`, preserved `workload.json` with exact token IDs, a flushed `requests.jsonl` journal, `requests.parquet`, `gpu_metrics.parquet`, `server_metrics.parquet`, `summary.json`, `report.md`, and a latency figure when data exists. Experiment reports include repetition medians/ranges and scaling curves when the matrix supports them. Failures and missing metrics stay visible. Interrupted runs finalize available observations; a hard process kill leaves the initial manifest and journal for inspection.

```sh
inference-lab compare BASELINE/summary.json CANDIDATE/summary.json --output reports/comparison.md
inference-lab check BASELINE/summary.json CANDIDATE/summary.json --policy configs/regression.yaml
```

Regression exit codes: `0` PASS, `1` FAIL, `2` INCONCLUSIVE. `run` returns `1` if any run is incomplete or any planned request fails, drops, or is canceled. Comparisons reject changed workload hashes, arrival policies, model revisions, hardware identity, measurement policies, or benchmark source. Serving flags may differ intentionally. Ratios with zero baselines or missing metrics are inconclusive. Compare matched repetitions; matrix aggregates do not become a single giant pseudo-replication.

## Tests and deployment

```sh
python -m pip install -e '.[dev]'
ruff check src tests
ruff format --check src tests
pytest
# Explicitly opt in on the GPU host after editing the real config:
INFERENCE_LAB_GPU_CONFIG=configs/concurrency.yaml pytest -m gpu
```

```sh
docker compose --profile mock up -d mock prometheus grafana
docker compose run --rm --use-aliases bench run /app/configs/compose-mock.yaml
```

The Compose benchmark configs enable the optional client Prometheus endpoint on port 9108. `--use-aliases` makes the one-off benchmark container discoverable by Prometheus. For native runs, enable `metrics.exporter_port` and configure a reachable scrape target; the default exporter host is loopback. Raw request artifacts provide exact failure counts even when a short run falls between scrapes.

Grafana: `http://localhost:3000` (login defined by `.env`). Prometheus: `http://localhost:9090`. Services bind host ports to loopback. Docker/GPU execution is an additional validation step on the target host. CPU CI runs unit/integration tests, a real HTTP mock demo, configuration validation, report regeneration, and regression fixtures; it never launches a GPU benchmark.

See [architecture](docs/architecture.md), [measurement methodology](docs/methodology.md), [experiment instructions](docs/experiments.md), and [limitations](docs/limitations.md).
