from __future__ import annotations

import torch

from .common import DTYPES, benchmark_callable, linear_cost


SHAPES = [(256, 4096), (512, 2048), (1024, 1024), (2048, 512), (4096, 256)]


def run(values, device: str, precision: str, warmup: int, repeats: int) -> list[dict]:
    dtype = DTYPES[precision]
    rows = []
    for output_width, input_width in SHAPES:
        x = torch.randn(1, input_width, device=device, dtype=dtype)
        weight = torch.randn(output_width, input_width, device=device, dtype=dtype)
        timing = benchmark_callable(lambda: x @ weight.T, device, warmup, repeats)
        cost = linear_cost(1, input_width, output_width, precision)
        rows.append({
            "experiment": "shape", "device": device, "precision": precision,
            "shape": f"{output_width}x{input_width}",
            "aspect_ratio": output_width / input_width,
            "input_width": input_width, "output_width": output_width,
            **cost, **timing,
            "measured_gflops": cost["flops"] / (timing["latency_mean_ms"] * 1e6),
        })
    return rows

