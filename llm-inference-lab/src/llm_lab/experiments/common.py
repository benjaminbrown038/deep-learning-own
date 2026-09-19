from __future__ import annotations

import csv
import statistics
import time
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import torch

from ..hardware import accelerator, synchronize


DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
TYPE_BYTES = {"fp32": 4.0, "fp16": 2.0, "bf16": 2.0, "int8": 1.0, "int4": 0.5}


def experiment_device(requested: str = "auto") -> str:
    return accelerator() if requested == "auto" else requested


def benchmark_callable(
    operation: Callable[[], object], device: str, warmup: int, repeats: int,
) -> dict:
    if warmup < 0 or repeats < 1:
        raise ValueError("warmup must be nonnegative and repeats must be positive")
    for _ in range(warmup):
        operation()
    synchronize(device)
    latencies = []
    for _ in range(repeats):
        synchronize(device)
        start = time.perf_counter()
        operation()
        synchronize(device)
        latencies.append((time.perf_counter() - start) * 1000)
    ordered = sorted(latencies)
    return {
        "latency_mean_ms": statistics.fmean(latencies),
        "latency_median_ms": statistics.median(latencies),
        "latency_min_ms": ordered[0],
        "latency_p95_ms": ordered[round((len(ordered) - 1) * 0.95)],
    }


def save_rows(rows: list[dict], path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return output


def plot_metrics(
    rows: list[dict], x: str, metrics: list[str], output: str | Path, title: str,
) -> Path:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(len(metrics), 1, figsize=(7.5, 3.7 * len(metrics)), squeeze=False)
    for axis, metric in zip(axes[:, 0], metrics):
        axis.plot([row[x] for row in rows], [row[metric] for row in rows], marker="o")
        axis.set_xlabel(x.replace("_", " ").title())
        axis.set_ylabel(metric.replace("_", " ").title())
        axis.grid(True, alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def linear_cost(batch: int, input_width: int, output_width: int, precision: str) -> dict:
    element_bytes = TYPE_BYTES[precision]
    parameters = input_width * output_width
    flops = 2 * batch * parameters
    bytes_moved = (
        parameters + batch * input_width + batch * output_width
    ) * element_bytes
    return {
        "parameters": parameters,
        "flops": flops,
        "bytes_moved": bytes_moved,
        "arithmetic_intensity": flops / bytes_moved,
    }

