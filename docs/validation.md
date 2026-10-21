# Validation record

The implementation was validated on an Apple Silicon macOS host using Python 3.11.5 and a Linux ARM64 Docker container using Python 3.11.14. **All inference observations below come from the explicitly identified mock server. No NVIDIA inference performance has been measured.**

| Check | Observed result |
| --- | --- |
| Native `pytest -q` | 75 passed, 1 skipped (the opt-in real-GPU test) |
| Linux container `python -m pytest -q` | 75 passed, 1 skipped |
| `ruff check src tests scripts` | Passed |
| `ruff format --check src tests scripts` | Passed; 28 files formatted |
| `python -m compileall -q src tests scripts` | Passed |
| `python -m pip check` | No broken requirements |
| Mock and GPU `docker compose … config --quiet` | Both configurations validate |
| `docker build -t inference-lab:local .` | Succeeded, including tokenizer and NVML client extras |
| Container mock HTTP demo | Succeeded; server startup, streaming, artifacts, figures and clean shutdown |
| Pinned real Qwen tokenizer in CPU container | Loaded revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`; verified three exact 512-token prompts; no model weights loaded |
| Final native mock concurrency matrix | Nine completed runs, 216/216 measured requests successful, one shared workload hash and source hash |
| Raw-artifact report regeneration and regression CLI | Passed, including PASS/FAIL/INCONCLUSIVE unit fixtures and strict comparability gates |
| Live open-loop overload and client metrics HTTP smoke check | 24 planned requests retained: 3 successful, 21 explicit queue drops; 30 successful exporter scrapes |

The final native demo is preserved at [results/mock-concurrency-7121b4c76d/report.md](../results/mock-concurrency-7121b4c76d/report.md). Its run subdirectories contain manifests, exact workloads, raw Parquet observations, journals and individual reports. The overload check is at [results/mock-openloop-smoke-a40bdaf9d7/report.md](../results/mock-openloop-smoke-a40bdaf9d7/report.md), with a captured `exporter-scrape.txt`. These directories are intentionally ignored by Git; explicitly select artifacts for publication after reviewing them.

The tests cover configuration validation, matrix expansion, actual length accounting, fixed seeds, mixed request classes, worker concurrency, constant/Poisson arrivals, overload, cancellation, streaming event fragmentation, role-only frames, multi-token chunks, one-chunk TPOT, missing/invalid usage, request deadlines, empty/incomplete/invalid responses, rate limits, synthetic OOMs, percentiles, failure denominators, SLO goodput, optional GPU capability fields, NVML failure cleanup, required-GPU sample failure, client metrics export, partial artifacts, readiness failures, raw report regeneration, all implemented figure types and regression decisions.

The workflow in `.github/workflows/ci.yml` is configured for CPU validation. It has not been executed on GitHub from this workspace. Docker Compose service startup for vLLM/DCGM and a real-GPU integration test remain target-host validation steps. Real measured optimization, task quality analysis, and the empirical GPU case study remain outstanding as described in the [proposal review](proposal-review.md).

To repeat the local validation:

```sh
source .venv/bin/activate
pytest -q
ruff check src tests scripts
ruff format --check src tests scripts
inference-lab demo --output results
python scripts/validate_artifacts.py results
docker compose --profile mock config --quiet
docker compose --profile gpu config --quiet
docker build -t inference-lab:local .
docker run --rm inference-lab:local demo
```
