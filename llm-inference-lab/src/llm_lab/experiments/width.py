from __future__ import annotations

import torch

from .common import DTYPES, benchmark_callable, linear_cost


def run(values, device: str, precision: str, warmup: int, repeats: int) -> list[dict]:
    dtype = DTYPES[precision]
    rows = []
    for width in values or [32, 64, 128, 256, 512, 1024, 2048]:
        x = torch.randn(1, width, device=device, dtype=dtype)
        weight = torch.randn(width, width, device=device, dtype=dtype)
        timing = benchmark_callable(lambda: x @ weight.T, device, warmup, repeats)
        cost = linear_cost(1, width, width, precision)
        rows.append({
            "experiment": "width", "device": device, "precision": precision,
            "width": width, **cost, **timing,
            "measured_gflops": cost["flops"] / (timing["latency_mean_ms"] * 1e6),
        })
    return rows

