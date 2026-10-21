import httpx
import pytest

from inference_lab.telemetry import Telemetry, parse_prometheus


def test_labels_and_histogram_infinity():
    rows = parse_prometheus(
        'vllm:num_requests_running{model_name="test"} 3\nlatency_bucket{le="+Inf"} 4\ninvalid NaN\n',
        5,
    )
    assert len(rows) == 2
    assert rows[0]["labels"] == '{"model_name": "test"}'
    assert rows[0]["value"] == 3
    assert rows[0]["timestamp_ms"] == 5


async def test_unavailable_server_metrics_are_explicit(config):
    import time

    config.metrics.collect_server = True

    async def handler(request):
        return httpx.Response(404)

    async with httpx.AsyncClient(
        base_url="http://test", transport=httpx.MockTransport(handler)
    ) as client:
        telemetry = Telemetry(config.metrics, client, time.perf_counter())
        await telemetry.sample()
    assert telemetry.errors[0]["source"] == "server"
    assert telemetry.summary()["server_samples"] == 0


@pytest.mark.parametrize("required", [False, True])
async def test_nvml_unavailable_cleanup_and_policy(config, monkeypatch, required):
    import time

    class MissingGPU:
        closed = False

        def __init__(self, indices):
            pass

        def start(self):
            raise RuntimeError("NVML missing")

        def close(self):
            MissingGPU.closed = True

    config.metrics.collect_gpu = True
    config.metrics.require_gpu = required
    monkeypatch.setattr("inference_lab.telemetry.GPUMonitor", MissingGPU)
    telemetry = Telemetry(config.metrics, None, time.perf_counter())
    if required:
        with pytest.raises(RuntimeError, match="required"):
            await telemetry.start()
    else:
        await telemetry.start()
        await telemetry.stop()
    assert MissingGPU.closed
    assert telemetry.summary()["gpu_samples"] == 0
    assert telemetry.errors[0]["source"] == "gpu"


def test_gpu_capabilities_and_multiple_devices(monkeypatch):
    import sys
    from types import SimpleNamespace

    from inference_lab.gpu_monitor import GPUMonitor

    class Unsupported(Exception):
        pass

    def no_power(handle):
        raise Unsupported()

    fake = SimpleNamespace(
        nvmlInit=lambda: None,
        nvmlShutdown=lambda: None,
        nvmlDeviceGetHandleByIndex=lambda i: i,
        nvmlDeviceGetName=lambda h: f"GPU-{h}",
        nvmlDeviceGetUUID=lambda h: f"UUID-{h}",
        nvmlSystemGetDriverVersion=lambda: "test-driver",
        nvmlDeviceGetMemoryInfo=lambda h: SimpleNamespace(total=1024, used=100 + h),
        nvmlDeviceGetUtilizationRates=lambda h: SimpleNamespace(gpu=10 + h, memory=20),
        nvmlDeviceGetPowerUsage=no_power,
        NVMLError=Unsupported,
        nvmlDeviceGetTemperature=lambda h, sensor: 40,
        NVML_TEMPERATURE_GPU=0,
    )
    monkeypatch.setitem(sys.modules, "pynvml", fake)
    gpu = GPUMonitor([0, 1])
    assert len(gpu.start()) == 2
    rows = gpu.sample(100)
    assert [row["memory_used_bytes"] for row in rows] == [100, 101]
    assert rows[0]["power_watts"] is None
    gpu.close()
    gpu.close()  # Shutdown is idempotent after successful initialization.
