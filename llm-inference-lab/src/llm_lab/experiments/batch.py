from __future__ import annotations

import torch

from .common import DTYPES, TYPE_BYTES, benchmark_callable, linear_cost


def run(values, device: str, precision: str, warmup: int, repeats: int, width: int = 1024) -> list[dict]:
    dtype = DTYPES[precision]
    weight = torch.randn(width, width, device=device, dtype=dtype)
    rows = []
    for batch_size in values or [1, 2, 4, 8, 16, 32]:
        x = torch.randn(batch_size, width, device=device, dtype=dtype)
        timing = benchmark_callable(lambda: x @ weight.T, device, warmup, repeats)
        cost = linear_cost(batch_size, width, width, precision)
        seconds = timing["latency_mean_ms"] / 1000
        rows.append({
            "experiment": "batch", "device": device, "precision": precision,
            "width": width, "batch_size": batch_size, **cost, **timing,
            "examples_per_second": batch_size / seconds,
            "weight_bytes_per_example": (
                width * width * TYPE_BYTES[precision] / batch_size
            ),
        })
    return rows
