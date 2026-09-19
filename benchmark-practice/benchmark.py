"""Benchmark one matrix operation. Start with: python benchmark.py

No model downloads, editable-package setup, or llm-lab command needed.
Requires PyTorch; everything else comes with Python.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import statistics
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def nonnegative_int(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return number


def add_settings(parser):
    parser.add_argument("--width", type=positive_int, default=512)
    parser.add_argument("--batch", type=positive_int, default=1)
    parser.add_argument("--dtype", choices=["fp32", "fp16"], default="fp32")
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
    parser.add_argument("--warmup", type=nonnegative_int, default=5)
    parser.add_argument("--repeats", type=positive_int, default=5)
    parser.add_argument("--iterations", type=positive_int, default=50,
                        help="matrix calls in each timed sample")
    parser.add_argument("--threads", type=positive_int, default=1,
                        help="PyTorch CPU intra-operation threads")
    parser.add_argument("--seed", type=nonnegative_int, default=0)
    parser.add_argument("--output", type=Path, default=Path("results"))


def matrix_cost(width, batch, bytes_per_value):
    """Logical tensor sizes; NOT measured memory traffic or peak RAM."""
    if min(width, batch, bytes_per_value) <= 0:
        raise ValueError("dimensions and element size must be positive")
    weights = width * width
    weight_bytes = weights * bytes_per_value
    input_bytes = batch * width * bytes_per_value
    output_bytes = batch * width * bytes_per_value
    logical_bytes = weight_bytes + input_bytes + output_bytes
    return {
        "weight_values": weights,
        "flops_per_call_approx": 2 * batch * weights,
        "weight_bytes": weight_bytes,
        "input_bytes": input_bytes,
        "output_bytes": output_bytes,
        "logical_tensor_bytes": logical_bytes,
        "ideal_flops_per_logical_byte": 2 * batch * weights / logical_bytes,
    }


def synchronize(torch, device):
    if device == "cuda":
        torch.cuda.synchronize(0)
    elif device == "mps":
        torch.mps.synchronize()


def run_case(args):
    # Import here so --help and the math tests work even without PyTorch.
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("Install the dependency: python -m pip install -r requirements.txt") from error

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Use --device cpu (or mps on a supported Mac).")
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable. Start with --device cpu.")
    if min(args.width, args.batch, args.repeats, args.iterations, args.threads) < 1 or args.warmup < 0:
        raise ValueError("dimensions, repeats, iterations and threads must be positive")

    torch.set_num_threads(args.threads)
    if args.device == "cuda":
        torch.cuda.set_device(0)
        torch.backends.cuda.matmul.allow_tf32 = False
    dtype = {"fp32": torch.float32, "fp16": torch.float16}[args.dtype]
    generator = torch.Generator(device="cpu").manual_seed(args.seed)

    # 1. Create data and move it BEFORE the clock starts.
    # X is [batch, width], W is [width, width], Y is [batch, width].
    # Generate W first so changing batch does not change the weight values.
    w = (torch.randn(args.width, args.width, generator=generator) / math.sqrt(args.width)).to(args.device, dtype)
    x = torch.randn(args.batch, args.width, generator=generator).to(args.device, dtype)
    y = torch.empty(args.batch, args.width, device=args.device, dtype=dtype)
    wt = w.T  # A transposed view, not a second copy of the weight matrix.

    samples = []
    with torch.inference_mode():
        # 2. Warm up. The output buffer is reused, not allocated inside the timer.
        for _ in range(args.warmup):
            torch.mm(x, wt, out=y)
        synchronize(torch, args.device)

        # 3. Measure repeated groups of identical operations.
        for sample in range(1, args.repeats + 1):
            synchronize(torch, args.device)
            start = time.perf_counter()
            for _ in range(args.iterations):
                torch.mm(x, wt, out=y)
            synchronize(torch, args.device)  # Wait for GPU completion before stopping.
            seconds = time.perf_counter() - start
            samples.append({"sample": sample, "iterations": args.iterations,
                            "sample_seconds": seconds,
                            "ms_per_call": seconds * 1000 / args.iterations})

        # 4. Sanity-check up to eight output columns against CPU float64.
        # This validation and transfer are deliberately OUTSIDE the timing.
        columns = min(8, args.width)
        reference = x.cpu().double() @ wt[:, :columns].cpu().double()
        observed = y[:, :columns].cpu().double()
        tolerance = 1e-2 if args.dtype == "fp16" else 1e-4
        torch.testing.assert_close(observed, reference, rtol=tolerance, atol=tolerance)
        max_error = (observed - reference).abs().max().item()

    costs = matrix_cost(args.width, args.batch, x.element_size())
    median_ms = statistics.median(sample["ms_per_call"] for sample in samples)
    device_name = platform.processor() or platform.machine()
    if args.device == "cuda":
        device_name = torch.cuda.get_device_name(0)
    elif args.device == "mps":
        device_name = "Apple MPS (exact chip not detected)"
    result = {
        "schema_version": 1,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "width": args.width, "batch": args.batch, "dtype": args.dtype,
        "device": args.device, "device_name": device_name,
        "python_version": platform.python_version(), "torch_version": str(torch.__version__),
        "platform": platform.platform(), "cpu_threads": torch.get_num_threads(),
        "torch_cuda_build": torch.version.cuda, "seed": args.seed,
        "warmup": args.warmup, "repeats": args.repeats, "iterations": args.iterations,
        "timing_method": "synchronized_wall_time_grouped_calls_preallocated_output",
        **costs,
        "median_ms_per_call": median_ms,
        "min_ms_per_call": min(sample["ms_per_call"] for sample in samples),
        "max_ms_per_call": max(sample["ms_per_call"] for sample in samples),
        "examples_per_second": args.batch / (median_ms / 1000),
        "gflops_per_second_approx": costs["flops_per_call_approx"] / (median_ms * 1e6),
        "checked_output_columns": columns,
        "check_max_absolute_error": max_error,
        "check_passed": True,
        "samples": samples,
    }
    return result


def print_result(result):
    print(f"\nY = X @ W.T | width={result['width']}, batch={result['batch']}")
    print(f"Device: {result['device']} | precision: {result['dtype']}")
    print(f"Weight values (not trained): {result['weight_values']:,}")
    print(f"Approximate FLOPs/call: {result['flops_per_call_approx']:,}")
    print(f"Weight storage: {result['weight_bytes']:,} bytes")
    print(f"X + W + Y logical storage: {result['logical_tensor_bytes']:,} bytes (not peak RAM)")
    print(f"Median time/call: {result['median_ms_per_call']:.6f} ms")
    print(f"Sample range: {result['min_ms_per_call']:.6f} to {result['max_ms_per_call']:.6f} ms/call")
    print(f"Throughput: {result['examples_per_second']:,.0f} examples/s (not tokens/s)")
    print("Output sanity check: passed")


def new_run_directory(output, label):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    path = Path(output) / f"{label}_{stamp}_{uuid.uuid4().hex[:8]}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def save_results(results, output, label):
    directory = new_run_directory(output, label)
    with (directory / "summary.csv").open("x", newline="", encoding="utf-8") as handle:
        fields = [key for key in results[0] if key != "samples"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: result[key] for key in fields} for result in results)
    with (directory / "details.json").open("x", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, allow_nan=False)
    print(f"\nSaved: {directory.resolve()}")
    return directory


def main():
    parser = argparse.ArgumentParser(description="Practice benchmarking a single matrix operation.")
    add_settings(parser)
    args = parser.parse_args()
    try:
        result = run_case(args)
        print_result(result)
        save_results([result], args.output, "single")
    except (RuntimeError, ValueError, AssertionError) as error:
        parser.exit(1, f"Benchmark failed: {error}\n")


if __name__ == "__main__":
    main()
