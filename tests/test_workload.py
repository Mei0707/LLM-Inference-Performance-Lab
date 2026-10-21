from inference_lab.config import Mix
from inference_lab.workload import MockTokenizer, generate, workload_hash


def test_exact_tokens_and_reproducibility(config):
    prompts = generate(config.workload, MockTokenizer(), 42)
    assert prompts == generate(config.workload, MockTokenizer(), 42)
    assert len(prompts) == config.workload.requests
    assert all(len(p.token_ids) == config.workload.input_tokens for p in prompts)
    assert workload_hash(prompts) != workload_hash(generate(config.workload, MockTokenizer(), 43))
    config.workload.concurrency = 8
    assert prompts == generate(config.workload, MockTokenizer(), 42)


def test_mixed_groups(config):
    config.workload.requests = 100
    config.workload.mixed = [
        Mix(weight=0.6, input_tokens=8, max_output_tokens=2, name="short"),
        Mix(weight=0.4, input_tokens=64, max_output_tokens=8, name="long"),
    ]
    prompts = generate(config.workload, MockTokenizer(), 1)
    assert {p.group for p in prompts} == {"short", "long"}
    assert all(p.input_tokens == (8 if p.group == "short" else 64) for p in prompts)
