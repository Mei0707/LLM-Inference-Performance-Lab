# Proposal review and implementation decisions

The 30-page proposal defines a useful inference measurement project. Its strongest choices are using an existing runtime, supporting both open and closed loops, preserving failures, and requiring raw observations behind every claim. Its 4–6 week timeline includes access to NVIDIA hardware and repeated scientific experiments; code implementation alone does not satisfy those empirical criteria.

The PDF is treated as a specification to review. Its roadmap, publication suggestions, and resume examples are not authorization to rent hardware, publish data, or claim unmeasured improvements.

| Review finding | Implementation response |
| --- | --- |
| “Token-length-controlled prompts” can be distorted by templates or retokenization | Load the pinned model tokenizer, generate exact token IDs, submit them to `/v1/completions`, preserve IDs and text, verify reported prompt usage. These are synthetic completion workloads, not a chat-quality evaluation. |
| Streaming chunks can contain multiple tokens | First nonempty content determines TTFT; final usage supplies generated-token counts. TPOT is explicitly a chunk-boundary estimate and is unavailable with one content chunk or absent usage. |
| Open loop needs a finite overload policy | Producer follows constant or seeded Poisson arrivals independently of responses; bounded workers and pending queue; excess arrivals become explicit `client_backlog_full` drops. Scheduled, dispatched and finished times are separate. |
| A timeout can erase queued or unstarted requests | Preallocate every planned observation, journal finished records, finalize pending and canceled requests in `finally`, preserve incomplete manifests and reports. |
| “Throughput” has ambiguous denominators | Use wall time from measured workload start through drain, excluding warm-up, report latency both from dispatch and scheduled arrival, count all planned requests in error/SLO denominators. |
| Server flags in YAML do not reconfigure vLLM | Variants target externally launched endpoints. `serve-command` renders declared configuration; `run --server NAME` supports sequential single-GPU experiments. |
| NVML is local to the client process | Require client execution on GPU host for GPU configs; separate client versions, operator runtime metadata, observed server data, and observed NVML hardware. |
| Runtime metric names change across versions | Preserve all Prometheus metric names and labels, record discovered names and scrape failures; recognize known KV-cache aliases without silently fabricating missing metrics. |
| Performance comparisons can be invalid | Match exact prompt hash, workload, model/tokenizer revisions, hardware identity, measurement policy and client source. Suppress ratios and return INCONCLUSIVE when incompatible, incomplete or missing. |
| Repeated requests are not independent experiments | Aggregate per-run metrics into medians/ranges across complete repetitions. Exclude incomplete repetitions and disclose counts. No fabricated confidence intervals. |
| Quantization is optional and hardware dependent | Provide a bitsandbytes template with the same model revision and tokenizer; leave execution and task-specific quality analysis to hardware validation. Preserve all outputs and truncation counts. |

## Acceptance status

Implemented: reusable package and CLI; strict YAML matrices; exact seeded workloads; mixed classes; open/closed loops; streaming and reliability measurements; readiness and warm-up; partial artifacts; JSON/Parquet storage; NVML and Prometheus collectors; reports and figures; comparisons and three-state regression checks; CPU mock server; unit/integration tests and opt-in GPU test; Docker/Compose; Prometheus/Grafana provisioning; methodology and experiment runbooks.

Verified locally: see [validation record](validation.md) for executed checks and artifacts. Mock data is prominently identified as simulation throughout manifests and reports.

Not yet empirically satisfied: real vLLM responses on NVIDIA hardware, measured GPU fit/utilization/memory, three completed real-GPU experiments, an evidence-backed serving improvement or negative finding, and the published GPU case study. The current Apple Silicon host cannot establish those outcomes. No GPU result is supplied or implied.

## Sources checked

The pinned example uses the [Qwen model revision](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/commit/989aa7980e4cf806f80c7fef2b1adb7bc71aa306). The deployment follows the versioned [vLLM Docker documentation](https://docs.vllm.ai/en/v0.12.0/deployment/docker/) and [OpenAI-compatible API documentation](https://docs.vllm.ai/en/v0.12.0/serving/openai_compatible_server/). Metric collection is based on the [v0.12 metrics design](https://docs.vllm.ai/en/v0.12.0/design/metrics/); raw names remain the source of truth for the installed runtime. GPU dashboard setup uses an [NVIDIA-published DCGM image](https://catalog.ngc.nvidia.com/orgs/nvidia/k8s/containers/dcgm-exporter/4.2.3-4.2.0-ubuntu22.04).
