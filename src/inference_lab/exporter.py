"""Optional live client metrics, including failures the inference server cannot observe."""

from prometheus_client import CollectorRegistry, Counter, Histogram, start_http_server


class BenchmarkExporter:
    def __init__(self):
        self.registry = CollectorRegistry()
        self.requests = Counter(
            "inference_lab_requests",
            "Finalized planned requests, including drops and cancellations",
            ["server", "status", "error_type"],
            registry=self.registry,
        )
        self.latency = Histogram(
            "inference_lab_schedule_latency_seconds",
            "Successful schedule-to-completion latency",
            ["server"],
            registry=self.registry,
        )
        self.ttft = Histogram(
            "inference_lab_ttft_seconds",
            "Successful client first-content latency",
            ["server"],
            registry=self.registry,
        )
        self.http_server = None
        self.thread = None

    def start(self, port, host):
        self.http_server, self.thread = start_http_server(port, addr=host, registry=self.registry)

    def observe(self, server, row):
        self.requests.labels(server, row["status"], row["error_type"] or "none").inc()
        if row["status"] == "success":
            if row["schedule_latency_ms"] is not None:
                self.latency.labels(server).observe(row["schedule_latency_ms"] / 1000)
            if row["ttft_ms"] is not None:
                self.ttft.labels(server).observe(row["ttft_ms"] / 1000)

    def close(self):
        if self.http_server:
            self.http_server.shutdown()
            self.http_server.server_close()
            self.thread.join(timeout=5)
            self.http_server = None
