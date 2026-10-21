"""Optional NVML collector with capability-aware fields, including failure reasons."""


class GPUMonitor:
    def __init__(self, indices):
        self.indices = indices
        self.nvml = None
        self.devices = []
        self.metadata = []
        self.initialized = False

    def start(self):
        import pynvml

        self.nvml = pynvml
        pynvml.nvmlInit()
        self.initialized = True
        for index in self.indices:
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            self.devices.append((index, handle))
            self.metadata.append(
                {
                    "index": index,
                    "name": pynvml.nvmlDeviceGetName(handle),
                    "uuid": pynvml.nvmlDeviceGetUUID(handle),
                    "memory_total_bytes": pynvml.nvmlDeviceGetMemoryInfo(handle).total,
                    "driver_version": pynvml.nvmlSystemGetDriverVersion(),
                }
            )
        return self.metadata

    def sample(self, timestamp_ms):
        rows = []
        n = self.nvml
        for index, handle in self.devices:
            row = {"timestamp_ms": timestamp_ms, "gpu_index": index}
            queries = {
                "utilization_percent": lambda handle=handle: (
                    n.nvmlDeviceGetUtilizationRates(handle).gpu
                ),
                "memory_utilization_percent": lambda handle=handle: (
                    n.nvmlDeviceGetUtilizationRates(handle).memory
                ),
                "memory_used_bytes": lambda handle=handle: n.nvmlDeviceGetMemoryInfo(handle).used,
                "memory_total_bytes": lambda handle=handle: n.nvmlDeviceGetMemoryInfo(handle).total,
                "power_watts": lambda handle=handle: n.nvmlDeviceGetPowerUsage(handle) / 1000,
                "temperature_celsius": lambda handle=handle: n.nvmlDeviceGetTemperature(
                    handle, n.NVML_TEMPERATURE_GPU
                ),
            }
            for field, query in queries.items():
                try:
                    row[field] = query()
                except n.NVMLError:
                    row[field] = None
            rows.append(row)
        return rows

    def close(self):
        if self.initialized:
            self.nvml.nvmlShutdown()
            self.initialized = False
