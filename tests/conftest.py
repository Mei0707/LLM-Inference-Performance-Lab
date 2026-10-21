import pytest

from inference_lab.cli import demo_config


@pytest.fixture
def config():
    c = demo_config("http://test")
    c.experiment.repetitions = 1
    c.experiment.warmup_requests = 0
    c.workload.requests = 6
    c.workload.concurrency = 2
    c.metrics.collect_server = False
    return c
