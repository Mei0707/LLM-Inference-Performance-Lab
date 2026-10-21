# Experiment runbook

First run `inference-lab demo`, then a real single-GPU smoke test. Install `.[tokenizer,gpu]` on the GPU host. Confirm GPU visibility and free memory, start the pinned vLLM image, record the resolved digest and actual CUDA/PyTorch/vLLM/driver versions, and populate `servers[].runtime`. Use `hardware_id` for an external server whose hardware cannot be observed by local NVML. Leave unobserved values null.

```sh
sh scripts/verify_gpu.sh
sh scripts/start_server.sh
docker compose exec vllm python -c 'import torch,vllm; print(torch.__version__,torch.version.cuda,vllm.__version__)'
docker image inspect vllm/vllm-openai:v0.12.0 --format '{{json .RepoDigests}}'
INFERENCE_LAB_GPU_CONFIG=configs/concurrency.yaml pytest -m gpu
```

| Experiment | Configuration | What to examine |
| --- | --- | --- |
| Concurrency | `configs/concurrency.yaml`: 1, 4, 8, three repetitions | Throughput plateau, p95 TTFT/latency, failure rate, SLO goodput and GPU activity |
| Prompt length | `configs/prompt-length.yaml`: 128/512/2048/4096 tokens | Prefill cost with fixed output cap and concurrency; check context fit first |
| Batching | `configs/batching.yaml`: 2048 vs 8192 batched-token budget | Throughput/tail latency tradeoff; queue and cache observations |
| Memory pressure | `configs/memory-pressure.yaml`: prompt length × active requests | Memory/KV cache, preemption, failures and saturation; total allocated GPU memory can stay flat with a preallocated cache |
| Quantization (optional) | `configs/quantization.yaml`: half vs bitsandbytes | Verify the selected vLLM image has the kernel/dependencies required by this GPU; compare memory and latency; inspect preserved outputs for degradation |
| Arrival rate | `configs/arrival-rate.yaml`: seeded Poisson rate sweep | Schedule latency, dispatch delay, drops and error rate; identify sustainable SLO goodput |
| Mixed | `configs/mixed.yaml`: short/medium/long 60/30/10 | Per-class p95 latency, fairness and head-of-line effects |

The matrices execute serially and reuse inputs for identical prompt/output definitions. For rigorous studies, vary configuration order across separate experiment invocations to reduce temporal drift. Three repetitions are an initial minimum; extend them when variation is large.

## Sequential variants on one GPU

The controller targets running endpoints. `serving_flags` document intent and generate commands; they do not reconfigure a server. Do not run two models concurrently on the same GPU and claim an isolated comparison.

```sh
inference-lab serve-command configs/batching.yaml --server small-batch
# Launch the printed command on the GPU host; it defaults to port 8000.
inference-lab run configs/batching.yaml --server small-batch
# Stop that server, wait for GPU memory to release.
inference-lab serve-command configs/batching.yaml --server large-batch
# Launch the printed command; it includes port 8001 from the variant endpoint.
inference-lab run configs/batching.yaml --server large-batch
```

Commands printed by `serve-command` carry model/tokenizer pins and serving flags; an explicit endpoint port is included automatically. If a proxy maps a different internal port, adjust the launch port and record the mapping. Record inspected command arguments. With a manually managed vLLM container, use the generated arguments after the image name (omit the `vllm serve` executable prefix because the official image already supplies its entry point).

Compare the baseline and candidate `summary.json` files only for matching workloads and hardware; inspect each repetition, not whichever single run happens to be fastest. Server variants in one config can also target separately controlled, equivalent GPU hosts, but strict hardware identity checks will suppress automated ratios across different devices.

## Interpretation

Use client curves alongside raw queue/cache/GPU time series. A throughput plateau by itself does not prove a compute bottleneck; memory reservation is not live KV occupancy; client dispatch delay can indicate a load-generator limit. Reports chart observed relationships but do not invent a bottleneck explanation. Use [case study template](case-study.md) to write the conclusion after real measurements exist. Quantization output quality requires a fixed task set and external evaluation beyond these synthetic prompts.
