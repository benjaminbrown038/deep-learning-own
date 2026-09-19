from __future__ import annotations

import torch
import torch.nn.functional as functional

from .common import DTYPES, TYPE_BYTES, benchmark_callable


def run(
    values, device: str, precision: str, warmup: int, repeats: int,
    heads: int = 12, head_dimension: int = 128,
) -> list[dict]:
    dtype = DTYPES[precision]
    rows = []
    for context_tokens in values or [128, 512, 1024, 2048]:
        query = torch.randn(1, heads, 1, head_dimension, device=device, dtype=dtype)
        key = torch.randn(1, heads, context_tokens, head_dimension, device=device, dtype=dtype)
        value = torch.randn_like(key)
        operation = lambda: functional.scaled_dot_product_attention(query, key, value)
        timing = benchmark_callable(operation, device, warmup, repeats)
        flops = 4 * heads * context_tokens * head_dimension
        kv_bytes = 2 * heads * context_tokens * head_dimension * TYPE_BYTES[precision]
        rows.append({
            "experiment": "context", "device": device, "precision": precision,
            "context_tokens": context_tokens, "heads": heads,
            "head_dimension": head_dimension, "attention_flops": flops,
            "kv_bytes_read": kv_bytes, **timing,
        })
    return rows

