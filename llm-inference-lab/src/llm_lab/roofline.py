from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from transformers import AutoConfig


WEIGHT_BYTES = {"fp32": 4.0, "fp16": 2.0, "bf16": 2.0, "int8": 1.05, "int4": 0.55}
ACTIVATION_BYTES = {"fp32": 4.0, "fp16": 2.0, "bf16": 2.0, "int8": 2.0, "int4": 2.0}


@dataclass
class OperationEstimate:
    operation: str
    flops: float
    bytes_moved: float
    kernel_launches: int
    arithmetic_intensity: float = 0.0
    attainable_tflops: float = 0.0
    roofline_time_us: float = 0.0
    launch_time_us: float = 0.0
    predicted_time_us: float = 0.0
    predicted_share_percent: float = 0.0
    bottleneck: str = ""


def decoder_operations(config, context_tokens: int, precision: str) -> list[OperationEstimate]:
    """Estimate one batch-one decode token for a dense gated-MLP decoder."""
    if context_tokens < 1:
        raise ValueError("context_tokens must be at least 1")
    if precision not in WEIGHT_BYTES:
        raise ValueError(f"unsupported precision: {precision}")

    layers = config.num_hidden_layers
    width = config.hidden_size
    heads = config.num_attention_heads
    kv_heads = getattr(config, "num_key_value_heads", heads)
    head_dim = width // heads
    kv_width = kv_heads * head_dim
    mlp = config.intermediate_size
    vocab = config.vocab_size
    wb = WEIGHT_BYTES[precision]
    ab = ACTIVATION_BYTES[precision]

    def linear(name: str, input_dim: int, output_dim: int, count: int = 1):
        weights = input_dim * output_dim * wb
        activations = (input_dim + output_dim) * ab
        return OperationEstimate(
            name,
            flops=2 * input_dim * output_dim * count,
            bytes_moved=(weights + activations) * count,
            kernel_launches=count,
        )

    operations = [
        linear("Q projection", width, width, layers),
        linear("K projection", width, kv_width, layers),
        linear("V projection", width, kv_width, layers),
        linear("Attention output projection", width, width, layers),
        OperationEstimate(
            "Attention QK + value mixing",
            flops=4 * layers * context_tokens * width,
            bytes_moved=layers * (2 * context_tokens * kv_width + 2 * width) * ab,
            kernel_launches=2 * layers,
        ),
        linear("MLP gate projection", width, mlp, layers),
        linear("MLP up projection", width, mlp, layers),
        linear("MLP down projection", mlp, width, layers),
        OperationEstimate(
            "RMSNorm + activation + residual",
            flops=layers * (20 * width + 4 * mlp),
            bytes_moved=layers * (10 * width + 4 * mlp) * ab,
            kernel_launches=5 * layers,
        ),
        linear("LM head", width, vocab),
    ]
    return operations


def analyze_operations(
    operations: list[OperationEstimate], peak_tflops: float,
    bandwidth_gbps: float, kernel_overhead_us: float,
) -> list[OperationEstimate]:
    if peak_tflops <= 0 or bandwidth_gbps <= 0 or kernel_overhead_us < 0:
        raise ValueError("hardware rates must be positive and overhead must be nonnegative")
    ridge = peak_tflops * 1000 / bandwidth_gbps
    for op in operations:
        op.arithmetic_intensity = op.flops / max(op.bytes_moved, 1)
        op.attainable_tflops = min(
            peak_tflops,
            op.arithmetic_intensity * bandwidth_gbps / 1000,
        )
        compute_us = op.flops / (peak_tflops * 1e12) * 1e6
        memory_us = op.bytes_moved / (bandwidth_gbps * 1e9) * 1e6
        op.roofline_time_us = max(compute_us, memory_us)
        op.launch_time_us = op.kernel_launches * kernel_overhead_us
        op.predicted_time_us = op.roofline_time_us + op.launch_time_us
        if op.launch_time_us > op.roofline_time_us:
            op.bottleneck = "kernel launch overhead"
        elif op.arithmetic_intensity < ridge:
            op.bottleneck = "memory bandwidth"
        else:
            op.bottleneck = "compute"
    total = sum(op.predicted_time_us for op in operations)
    for op in operations:
        op.predicted_share_percent = 100 * op.predicted_time_us / total
    return operations


def save_analysis(operations: list[OperationEstimate], csv_path: str | Path) -> Path:
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [asdict(op) for op in operations]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def plot_roofline(
    operations: list[OperationEstimate], peak_tflops: float,
    bandwidth_gbps: float, output_path: str | Path,
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    intensities = [10 ** (x / 20) for x in range(-40, 101)]
    roof = [min(peak_tflops, intensity * bandwidth_gbps / 1000) for intensity in intensities]
    fig, axis = plt.subplots(figsize=(8, 5.5))
    axis.loglog(intensities, roof, color="black", label="Hardware roofline")
    for op in operations:
        axis.scatter(op.arithmetic_intensity, op.attainable_tflops, s=55)
        axis.annotate(op.operation, (op.arithmetic_intensity, op.attainable_tflops), fontsize=7)
    axis.set_xlabel("Arithmetic intensity (FLOPs/byte)")
    axis.set_ylabel("Attainable throughput (TFLOP/s)")
    axis.set_title("Estimated decode-operation roofline")
    axis.grid(True, which="both", alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def build_roofline_report(
    model_name: str, precision: str, context_tokens: int,
    peak_tflops: float, bandwidth_gbps: float, kernel_overhead_us: float,
    csv_path: str, plot_path: str,
) -> list[OperationEstimate]:
    config = AutoConfig.from_pretrained(model_name)
    operations = decoder_operations(config, context_tokens, precision)
    analyze_operations(operations, peak_tflops, bandwidth_gbps, kernel_overhead_us)
    save_analysis(operations, csv_path)
    plot_roofline(operations, peak_tflops, bandwidth_gbps, plot_path)
    return operations


def print_analysis(operations: list[OperationEstimate]) -> None:
    print("\nPredicted one-token decode bottlenecks")
    print("-" * 96)
    print(f"{'Operation':36} {'AI':>9} {'Time us':>12} {'Share':>9}  Bottleneck")
    for op in sorted(operations, key=lambda item: item.predicted_time_us, reverse=True):
        print(
            f"{op.operation:36} {op.arithmetic_intensity:9.3f} "
            f"{op.predicted_time_us:12.2f} {op.predicted_share_percent:8.1f}%  "
            f"{op.bottleneck}"
        )

