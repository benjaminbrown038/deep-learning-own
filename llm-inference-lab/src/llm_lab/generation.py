from __future__ import annotations

import time

import torch

from .hardware import synchronize


def greedy_generate_timed(model, inputs, new_tokens: int, device: str) -> dict:
    """Run exact prefill plus one-token decode iterations with a KV cache."""
    if new_tokens < 1:
        raise ValueError("new_tokens must be at least 1")

    input_tokens = inputs["input_ids"].shape[1]
    attention_mask = inputs.get("attention_mask")

    synchronize(device)
    prefill_start = time.perf_counter()
    with torch.inference_mode():
        outputs = model(**inputs, use_cache=True, return_dict=True)
    synchronize(device)
    prefill_seconds = time.perf_counter() - prefill_start

    cache = outputs.past_key_values
    next_token = outputs.logits[:, -1:].argmax(dim=-1)
    generated = [next_token]
    token_latencies_ms: list[float] = []

    for _ in range(new_tokens - 1):
        if attention_mask is not None:
            attention_mask = torch.cat(
                [attention_mask, torch.ones_like(next_token)], dim=1
            )
        synchronize(device)
        token_start = time.perf_counter()
        with torch.inference_mode():
            outputs = model(
                input_ids=next_token,
                attention_mask=attention_mask,
                past_key_values=cache,
                use_cache=True,
                return_dict=True,
            )
        synchronize(device)
        token_latencies_ms.append((time.perf_counter() - token_start) * 1000)
        cache = outputs.past_key_values
        next_token = outputs.logits[:, -1:].argmax(dim=-1)
        generated.append(next_token)

    generated_ids = torch.cat(generated, dim=1)
    decode_seconds = sum(token_latencies_ms) / 1000
    decode_count = len(token_latencies_ms)
    ordered = sorted(token_latencies_ms)

    def percentile(p: float) -> float | None:
        if not ordered:
            return None
        index = round((len(ordered) - 1) * p)
        return ordered[index]

    return {
        "generated_ids": generated_ids,
        "input_tokens": input_tokens,
        "generated_tokens": generated_ids.shape[1],
        "prefill_seconds": prefill_seconds,
        "prefill_tokens_per_second": input_tokens / prefill_seconds,
        "time_to_first_token_ms": prefill_seconds * 1000,
        "decode_seconds": decode_seconds,
        "decode_tokens_per_second": (
            decode_count / decode_seconds if decode_seconds else None
        ),
        "decode_latency_mean_ms": (
            sum(ordered) / len(ordered) if ordered else None
        ),
        "decode_latency_p50_ms": percentile(0.50),
        "decode_latency_p95_ms": percentile(0.95),
        "decode_latency_p99_ms": percentile(0.99),
        "total_seconds": prefill_seconds + decode_seconds,
        "token_latencies_ms": token_latencies_ms,
    }

