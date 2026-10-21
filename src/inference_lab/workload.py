"""Generate and preserve exact token IDs; no chat template adds hidden input tokens."""

import hashlib
import json
import random
from dataclasses import asdict, dataclass

from inference_lab.config import Model, Workload

WORDS = "river forest mountain morning science history music garden winter travel bright calm explain describe compare discover planet story language city".split()


class MockTokenizer:
    """Offline protocol fixture only; never used for real-model measurements."""

    def encode(self, text, add_special_tokens=False):
        return [WORDS.index(word) + 1 for word in text.split()]

    def decode(self, ids):
        return " ".join(WORDS[(i - 1) % len(WORDS)] for i in ids)


def load_tokenizer(model: Model):
    if model.tokenizer == "mock":
        return MockTokenizer()
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Install inference-lab[tokenizer] for real-model workloads") from exc
    return AutoTokenizer.from_pretrained(
        model.tokenizer_id or model.id,
        revision=model.tokenizer_revision or model.revision,
        trust_remote_code=False,
    )


@dataclass(frozen=True)
class Prompt:
    request_id: str
    token_ids: list[int]
    text: str
    input_tokens: int
    max_output_tokens: int
    group: str = "uniform"

    def to_dict(self):
        return asdict(self)


def generate(workload: Workload, tokenizer, seed: int) -> list[Prompt]:
    rng = random.Random(seed)
    prompts = []
    for i in range(workload.requests):
        if workload.mixed:
            mix = rng.choices(workload.mixed, weights=[x.weight for x in workload.mixed])[0]
            length, output, group = mix.input_tokens, mix.max_output_tokens, mix.name
        else:
            length, output, group = workload.input_tokens, workload.max_output_tokens, "uniform"
        ids = []
        while len(ids) < length:
            text = " ".join(rng.choices(WORDS, k=max(32, length - len(ids))))
            encoded = tokenizer.encode(text, add_special_tokens=False)
            if not encoded:
                raise ValueError("Tokenizer produced no IDs")
            ids.extend(encoded)
        ids = ids[:length]
        prompts.append(Prompt(f"req-{i:06d}", ids, tokenizer.decode(ids), len(ids), output, group))
    return prompts


def workload_hash(prompts: list[Prompt]) -> str:
    canonical = json.dumps([p.to_dict() for p in prompts], sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()
