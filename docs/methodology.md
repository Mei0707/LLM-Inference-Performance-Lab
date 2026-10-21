# Measurement methodology

## Workload control

Real experiments require immutable 40-character Hugging Face revision SHAs. The actual tokenizer is loaded without remote code execution. Synthetic natural-language vocabulary sequences are tokenized and truncated to exactly the requested length. Raw IDs go directly to the completions endpoint, preventing hidden template or special-token additions. The workload preserves decoded text, IDs, class, output limit and a SHA256 hash. Actual server prompt counts, when available, must match. A different seed changes inputs; concurrency and repetition do not. This is a synthetic systems workload, not realistic task quality data.

Warm-up uses a separate seed and results file, runs before telemetry, and contributes no measured throughput or latency. The default example disables prefix caching so shared prompt prefixes and repetition order do not silently become cache experiments. Keep cache behavior and generation defaults explicit. `sampling_seed`, `temperature`, `ignore_eos` and requested output limits are recorded. `ignore_eos: true` is available for controlled fixed-length decode experiments; otherwise actual generation can end early. Truncation and finish reason are preserved.

Closed loop continuously feeds a fixed worker count. Its scheduled time is when a worker picks up the request, so client-side preload does not masquerade as open-loop arrival delay. Open loop schedules independent constant or Poisson arrivals from a monotonic origin, dispatches through the configured concurrency cap, and retains scheduling delay. `max_pending` limits the client queue. A full queue causes an explicit dropped request; overload is not hidden by slowing the arrival producer. The first arrival is at zero; subsequent intervals follow the configured distribution. The entire planned workload is finite and bounded by request count and run deadline.

## Timings and accounting

All request timings are milliseconds relative to the same `perf_counter` origin; UTC timestamps anchor the manifest. TTFT starts at dispatch immediately before the HTTP request and stops on first nonempty generated content. Headers, comments and role-only events are ignored. This measures delivery of content, which can lag server generation for token pieces that decode to no text.

Service latency ends when the stream's `[DONE]` marker is consumed; schedule latency starts at the planned arrival. HTTP stream termination without `[DONE]` or a finish reason is incomplete. Final usage frames contain no content and do not move the last-content timestamp. Transport failure after partial generation preserves the partial text and timings.

The reported TPOT estimate is `(last content timestamp - first content timestamp) / (completion tokens - 1)`. It requires at least two tokens, two content events, and verified final server usage. It is not exact per-token latency when chunks contain multiple tokens. Raw content event counts and timestamps document that limit. Missing server usage is never replaced by character counts, event counts or the requested output cap.

Throughput uses the full workload wall interval including request launch, scheduled arrival gaps, and drain of slow requests. Telemetry initialization/shutdown, readiness, warm-up, tokenization and report rendering are excluded. Requests/s and input/output tokens/s count successful requests; input throughput is labeled separately. Output tokens/s is unavailable if any successful response lacks usage. Partial tokens from failed requests remain in raw observations but do not inflate successful throughput. Error rate includes every planned failure, timeout, cancellation and client drop. SLO goodput counts successful requests satisfying configured TTFT and schedule-to-completion limits; failed requests stay in the SLO denominator.

Percentiles use sorted linear interpolation. Missing observations are excluded from the latency distribution with sample counts retained; reliability is reported separately. p99 is suppressed below `p99_min_samples` (default 100, a disclosure threshold rather than statistical certainty). Repetition medians/ranges are descriptive; no independence or confidence claim is made about thousands of requests sharing one GPU run.

## Telemetry and comparisons

NVML fields use explicit units: bytes, percent, watts and Celsius. Unsupported optional device fields are null. Memory-utilization percent is NVML's memory activity indicator, not a measured bandwidth in GB/s. Sampling interval and errors are recorded. Prometheus samples retain timestamps, names and labels, including histograms/counters. Histogram quantiles are calculated by Prometheus/Grafana; client tail latency stays independent. Finite `+Inf` bucket counts are retained; nonfinite numerical observations are omitted.

GPU rows describe local devices only. Remote clients must use separately exported GPU observations or run the benchmark on the GPU machine. Runtime versions in `server.runtime` are operator-supplied; `client_package_versions` must never be mistaken for server versions. Missing runtime fields stay null.

Comparison requires matching hardware identity, exact workload, prompt hash, model/tokenizer revisions, measurement policy, and benchmark source hash. Intended serving flags, precision and runtime versions can change; preserve manifests to explain that change. Different model weights produce a qualified observational report without improvement ratios. Baseline zero, absent metrics and incomplete observations make ratio-based regressions inconclusive. For changing benchmark code, regenerate baseline and candidate with the same version before checking.
