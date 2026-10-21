# Real-GPU case study worksheet

Status: **awaiting real GPU measurements**. This is a worksheet, not a performance result.

1. Identify the exact GPU/VRAM, device UUID or hardware identity, driver, CUDA, PyTorch, vLLM, container digest, model/tokenizer revision, source commit/hash and all serving flags.
2. Link the preserved workload hash, request count, seed, arrival policy, input distribution, output caps, cache behavior, warm-up policy, sampling interval and SLO.
3. Run concurrency, prompt-length and batching studies with at least three repetitions per important configuration. Link raw artifacts and regenerate charts from them.
4. Select a baseline and one changed serving parameter. Explain the hypothesis before examining candidate performance. Keep hardware and workload fixed.
5. Report per-run throughput, p95 TTFT/latency, SLO goodput, errors, generated-token counts and GPU/KV-cache observations. Show median and observed range across repetitions.
6. Link the strict comparison and regression outputs. Explain missing or incompatible evidence and observed variation.
7. State a conclusion supported by the data: a measured benefit, a tradeoff, no clear improvement, or a regression. Correlate queue/cache/GPU evidence without asserting causality that was not tested.
8. For quantization, include the fixed quality sample, evaluator and actual quality results. Synthetic performance prompts alone cannot establish answer quality.

Do not replace these fields with mock data or predicted percentages. A reviewer should be able to follow every numerical claim to raw measurements.
