from __future__ import annotations

import torch

from .common import DTYPES, benchmark_callable, linear_cost


def run(values, device: str, precision: str, warmup: int, repeats: int, width: int = 512) -> list[dict]:
    dtype = DTYPES[precision]
    x = torch.randn(1, width, device=device, dtype=dtype)
    weight = torch.randn(width, width, device=device, dtype=dtype)
    rows = []
    for layers in values or [1, 2, 4, 8, 16, 32]:
        def operation():
            state = x
            for _ in range(layers):
                state = torch.relu(state @ weight.T)
            return state
        timing = benchmark_callable(operation, device, warmup, repeats)
        one = linear_cost(1, width, width, precision)
        rows.append({
            "experiment": "depth", "device": device, "precision": precision,
            "width": width, "layers": layers,
            "parameters_if_unique": one["parameters"] * layers,
            "flops": one["flops"] * layers,
            "bytes_moved": one["bytes_moved"] * layers,
            **timing,
        })
    return rows

