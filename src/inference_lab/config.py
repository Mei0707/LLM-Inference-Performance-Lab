"""Strict experiment schemas. Lists describe an explicit workload Cartesian product."""

import itertools
import re
from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

PositiveInt = Annotated[int, Field(gt=0)]
PositiveFloat = Annotated[float, Field(gt=0, allow_inf_nan=False)]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Experiment(StrictModel):
    name: str = Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_-]*$")
    repetitions: PositiveInt = 3
    warmup_requests: int = Field(default=5, ge=0)
    random_seed: int = 42
    timeout_seconds: PositiveFloat = 600


class Model(StrictModel):
    id: str
    revision: str
    tokenizer_id: str | None = None
    tokenizer_revision: str | None = None
    tokenizer: Literal["huggingface", "mock"] = "huggingface"
    dtype: str = "auto"
    quantization: str | None = None
    max_model_len: PositiveInt = 4096

    @model_validator(mode="after")
    def pinned(self):
        if self.tokenizer == "huggingface":
            for revision in [self.revision, self.tokenizer_revision or self.revision]:
                if not re.fullmatch(r"[0-9a-fA-F]{40}", revision):
                    raise ValueError(
                        "Real models/tokenizers require an immutable 40-character revision SHA"
                    )
        return self


class Runtime(StrictModel):
    """Operator supplied observations; distinct from locally discovered versions."""

    container_image: str | None = None
    container_digest: str | None = None
    vllm_version: str | None = None
    pytorch_version: str | None = None
    cuda_version: str | None = None
    gpu_model: str | None = None
    gpu_memory_bytes: PositiveInt | None = None
    driver_version: str | None = None
    hardware_id: str | None = None
    notes: str | None = None


class Server(StrictModel):
    name: str = Field(default="baseline", pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_-]*$")
    base_url: str = "http://127.0.0.1:8000"
    api_key_env: str | None = None
    readiness_timeout_seconds: PositiveFloat = 120
    request_timeout_seconds: PositiveFloat = 60
    serving_flags: dict[str, str | int | float | bool] = Field(default_factory=dict)
    runtime: Runtime = Field(default_factory=Runtime)
    model: Model | None = None

    @model_validator(mode="after")
    def root_url(self):
        from urllib.parse import urlsplit

        parts = urlsplit(self.base_url)
        if parts.scheme not in {"http", "https"} or not parts.netloc:
            raise ValueError("base_url must be an http(s) origin")
        if parts.username or parts.password or parts.query or parts.fragment:
            raise ValueError(
                "Use api_key_env for secrets; URL credentials/query/fragment are forbidden"
            )
        if parts.path.rstrip("/"):
            raise ValueError("base_url is the server origin, without /v1")
        self.base_url = self.base_url.rstrip("/")
        return self


class Mix(StrictModel):
    weight: PositiveFloat
    input_tokens: PositiveInt
    max_output_tokens: PositiveInt
    name: str


class Workload(StrictModel):
    mode: Literal["closed_loop", "open_loop"] = "closed_loop"
    requests: PositiveInt = 100
    input_tokens: PositiveInt | list[PositiveInt] = 128
    max_output_tokens: PositiveInt | list[PositiveInt] = 64
    concurrency: PositiveInt | list[PositiveInt] = 1
    arrival_rate: PositiveFloat | list[PositiveFloat] | None = None
    arrival_distribution: Literal["constant", "poisson"] = "constant"
    max_pending: PositiveInt = 1000
    temperature: float = Field(default=0, ge=0, le=2)
    sampling_seed: int = 0
    streaming: Literal[True] = True
    ignore_eos: bool = False
    mixed: list[Mix] = Field(default_factory=list)

    @model_validator(mode="after")
    def dimensions(self):
        for name in ["input_tokens", "max_output_tokens", "concurrency", "arrival_rate"]:
            value = getattr(self, name)
            if isinstance(value, list) and not value:
                raise ValueError(f"{name} cannot be empty")
        if self.mode == "open_loop" and self.arrival_rate is None:
            raise ValueError("open_loop requires arrival_rate")
        if self.mode == "closed_loop" and self.arrival_rate is not None:
            raise ValueError("arrival_rate only applies to open_loop")
        return self


class Metrics(StrictModel):
    collect_gpu: bool = True
    collect_server: bool = True
    require_gpu: bool = False
    sample_interval_ms: int = Field(default=500, ge=10)
    exporter_port: int | None = Field(default=None, ge=1, le=65535)
    exporter_host: str = "127.0.0.1"
    gpu_indices: list[Annotated[int, Field(ge=0)]] = Field(
        default_factory=lambda: [0], min_length=1
    )


class Report(StrictModel):
    percentiles: list[float] = Field(default_factory=lambda: [50, 95, 99])
    p99_min_samples: PositiveInt = 100
    slo_ttft_ms: PositiveFloat | None = None
    slo_latency_ms: PositiveFloat | None = None

    @model_validator(mode="after")
    def percentiles_valid(self):
        if any(p <= 0 or p >= 100 for p in self.percentiles):
            raise ValueError("percentiles must be between 0 and 100")
        return self


class Config(StrictModel):
    experiment: Experiment
    model: Model
    workload: Workload
    servers: list[Server] = Field(default_factory=lambda: [Server()])
    metrics: Metrics = Field(default_factory=Metrics)
    report: Report = Field(default_factory=Report)

    @model_validator(mode="after")
    def context(self):
        if not self.servers or len({s.name for s in self.servers}) != len(self.servers):
            raise ValueError("servers must be nonempty with unique names")
        if self.metrics.require_gpu and not self.metrics.collect_gpu:
            raise ValueError("require_gpu needs collect_gpu")
        w = self.workload
        sizes = (
            [x.input_tokens + x.max_output_tokens for x in w.mixed]
            if w.mixed
            else [i + o for i in as_list(w.input_tokens) for o in as_list(w.max_output_tokens)]
        )
        for server in self.servers:
            model = server.model or self.model
            if max(sizes) > model.max_model_len:
                raise ValueError(
                    f"Input + requested output exceeds model context for {server.name}"
                )
        return self

    def runs(self):
        keys = ["input_tokens", "max_output_tokens", "concurrency", "arrival_rate"]
        for server in self.servers:
            for values in itertools.product(*(as_list(getattr(self.workload, k)) for k in keys)):
                workload = self.workload.model_copy(update=dict(zip(keys, values, strict=True)))
                for repetition in range(self.experiment.repetitions):
                    yield server, workload, repetition


def as_list(value):
    return value if isinstance(value, list) else [value]


def load_config(path: str | Path) -> Config:
    return Config.model_validate(yaml.safe_load(Path(path).read_text()))
