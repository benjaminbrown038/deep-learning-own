from __future__ import annotations


BYTES_PER_VALUE = {
    "fp32": 4.0,
    "fp16": 2.0,
    "bf16": 2.0,
    "int8": 1.0,
    "int4": 0.5,
}


def linear_memory(rows: int, columns: int, precision: str, batch_size: int = 1) -> dict:
    """Count ideal bytes read/written by one dense linear operation."""
    if precision not in BYTES_PER_VALUE:
        raise ValueError(f"unsupported precision: {precision}")
    if min(rows, columns, batch_size) < 1:
        raise ValueError("all dimensions must be positive")
    element_bytes = BYTES_PER_VALUE[precision]
    weight_bytes = rows * columns * element_bytes
    input_bytes = batch_size * columns * element_bytes
    output_bytes = batch_size * rows * element_bytes
    total = weight_bytes + input_bytes + output_bytes
    flops = 2 * batch_size * rows * columns
    return {
        "precision": precision,
        "weight_bytes": weight_bytes,
        "input_bytes": input_bytes,
        "output_bytes": output_bytes,
        "total_bytes": total,
        "flops": flops,
        "arithmetic_intensity_flops_per_byte": flops / total,
    }


def ideal_time_seconds(flops: float, bytes_moved: float, compute_flops: float, bandwidth_bps: float) -> dict:
    """Simple roofline lower bound: the slower of compute and memory wins."""
    compute_time = flops / compute_flops
    memory_time = bytes_moved / bandwidth_bps
    return {
        "compute_time_seconds": compute_time,
        "memory_time_seconds": memory_time,
        "ideal_time_seconds": max(compute_time, memory_time),
        "limiter": "compute" if compute_time > memory_time else "memory bandwidth",
    }

