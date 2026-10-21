import copy
from pathlib import Path

import pytest
from pydantic import ValidationError

from inference_lab.cli import serve_command
from inference_lab.config import Config, load_config


@pytest.mark.parametrize(
    "update",
    [
        {"requests": 0},
        {"concurrency": []},
        {"arrival_rate": 2},
        {"mode": "open_loop"},
        {"input_tokens": 10000},
        {"max_pending": 0},
        {"concurrency": [1, -1]},
        {"streaming": False},
        {"arrival_rate": float("nan")},
        {"mode": "open_loop", "arrival_rate": []},
        {"typo": 1},
    ],
)
def test_invalid_workloads(config, update):
    data = config.model_dump()
    data["workload"].update(update)
    with pytest.raises(ValidationError):
        Config.model_validate(data)


def test_real_model_requires_pin(config):
    data = config.model_dump()
    data["model"].update(tokenizer="huggingface", revision="main")
    with pytest.raises(ValidationError, match="immutable"):
        Config.model_validate(data)


def test_matrix(config):
    config.experiment.repetitions = 3
    config.workload.concurrency = [1, 4, 8]
    config.workload.input_tokens = [32, 64]
    runs = list(config.runs())
    assert len(runs) == 18
    assert all(isinstance(w.concurrency, int) for _, w, _ in runs)


@pytest.mark.parametrize(
    "url",
    [
        "http://user:secret@host",
        "http://host/v1",
        "file:///tmp/server",
        "http://host?api_key=secret",
    ],
)
def test_secrets_and_ambiguous_urls_rejected(config, url):
    data = config.model_dump()
    data["servers"][0]["base_url"] = url
    with pytest.raises(ValidationError):
        Config.model_validate(data)


def test_examples_validate():
    for path in Path("configs").glob("*.yaml"):
        if path.name != "regression.yaml":
            assert list(load_config(path).runs())


def test_serve_command_pins_and_does_not_override(config):
    data = copy.deepcopy(config.model_dump())
    data["model"].update(tokenizer="huggingface", revision="a" * 40)
    data["servers"][0]["serving_flags"] = {"max_num_seqs": 16}
    config = Config.model_validate(data)
    command = serve_command(config, "baseline")
    assert "--revision " + "a" * 40 in command
    assert "--max-num-seqs 16" in command
    config.servers[0].serving_flags["revision"] = "main"
    with pytest.raises(ValueError, match="conflicts"):
        serve_command(config, "baseline")
