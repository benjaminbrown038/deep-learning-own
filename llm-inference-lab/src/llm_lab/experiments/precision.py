from __future__ import annotations

import torch

from .common import DTYPES, TYPE_BYTES, benchmark_callable, linear_cost


def _symmetric_quantize(weight: torch.Tensor, bits: int):
    maximum = 2 ** (bits - 1) - 1
    scale = weight.abs().max() / maximum
    scale = torch.clamp(scale, min=torch.finfo(torch.float32).eps)
    quantized = torch.clamp(torch.round(weight / scale), -maximum, maximum).to(torch.int8)
    return quantized, scale


def run(values, device: str, precision: str, warmup: int, repeats: int, width: int = 1024) -> list[dict]:
    del values, precision
    baseline_weight = torch.randn(width, width, device=device, dtype=torch.float32)
    baseline_x = torch.randn(1, width, device=device, dtype=torch.float32)
    baseline_output = baseline_x @ baseline_weight.T
    rows = []
    for tested in ("fp32", "fp16", "int8", "int4"):
        simulated = tested in {"int8", "int4"}
        if simulated:
            bits = 8 if tested == "int8" else 4
            quantized, scale = _symmetric_quantize(baseline_weight, bits)
            execution_weight = quantized.float() * scale
            execution_x = baseline_x
            execution_dtype = "fp32 after dequantization"
        else:
            dtype = DTYPES[tested]
            execution_weight = baseline_weight.to(dtype)
            execution_x = baseline_x.to(dtype)
            execution_dtype = tested
        output = execution_x @ execution_weight.T
        error = torch.linalg.vector_norm(output.float() - baseline_output) / torch.linalg.vector_norm(baseline_output)
        timing = benchmark_callable(lambda: execution_x @ execution_weight.T, device, warmup, repeats)
        cost = linear_cost(1, width, width, tested)
        rows.append({
            "experiment": "precision", "device": device, "precision": tested,
            "width": width, "simulated_quantization": simulated,
            "execution_dtype": execution_dtype,
            "theoretical_weight_bytes": width * width * TYPE_BYTES[tested],
            "relative_l2_error": float(error), **cost, **timing,
        })
    return rows

