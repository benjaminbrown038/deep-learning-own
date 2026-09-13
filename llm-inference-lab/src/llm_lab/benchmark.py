from __future__ import annotations

from datetime import datetime, timezone

import torch

from .generation import greedy_generate_timed
from .hardware import collect_hardware, peak_memory_gb, reset_peak_memory
from .prompt import build_context_prompt, tokenize_chat


def benchmark_loaded_model(
    *, tokenizer, model, input_device, model_name: str, size: str,
    precision: str, prompt: str, context_tokens: int, new_tokens: int,
    device: str,
) -> dict:
    constructed_prompt = build_context_prompt(tokenizer, prompt, context_tokens)
    inputs = tokenize_chat(tokenizer, constructed_prompt, input_device)

    # Warm-up is intentionally excluded from reported measurements.
    with torch.inference_mode():
        model(**inputs, use_cache=True, return_dict=True)
    reset_peak_memory(device)

    timing = greedy_generate_timed(model, inputs, new_tokens, device)
    memory = peak_memory_gb(device)
    response = tokenizer.decode(timing.pop("generated_ids")[0], skip_special_tokens=True)

    result = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **collect_hardware(device),
        "model": model_name,
        "size": size,
        "precision": precision,
        "batch_size": inputs["input_ids"].shape[0],
        "requested_context_tokens": context_tokens,
        **timing,
        "peak_accelerator_memory_gb": memory,
        "response": response,
    }
    return result


def print_result(result: dict) -> None:
    print("\nResults")
    print(f"Input tokens: {result['input_tokens']}")
    print(f"Generated tokens: {result['generated_tokens']}")
    print(f"Prefill: {result['prefill_tokens_per_second']:.2f} tok/s")
    decode = result["decode_tokens_per_second"]
    print(f"Decode: {decode:.2f} tok/s" if decode else "Decode: n/a")
    print(f"Time to first token: {result['time_to_first_token_ms']:.2f} ms")
    print(f"Total time: {result['total_seconds']:.3f} s")
    if result["peak_accelerator_memory_gb"] is not None:
        print(f"Peak accelerator memory: {result['peak_accelerator_memory_gb']:.3f} GB")
    print(f"\nResponse:\n{result['response']}")

