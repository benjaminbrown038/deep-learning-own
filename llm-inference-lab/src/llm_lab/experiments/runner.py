from __future__ import annotations

from pathlib import Path

from . import batch, context, depth, precision, shape, width
from .common import experiment_device, plot_metrics, save_rows


MODULES = {
    "width": width,
    "shape": shape,
    "depth": depth,
    "batch": batch,
    "precision": precision,
    "context": context,
}

PLOTS = {
    "width": ("width", ["latency_mean_ms", "measured_gflops"]),
    "shape": ("aspect_ratio", ["latency_mean_ms", "measured_gflops"]),
    "depth": ("layers", ["latency_mean_ms", "flops"]),
    "batch": ("batch_size", ["latency_mean_ms", "examples_per_second"]),
    "precision": ("theoretical_weight_bytes", ["latency_mean_ms", "relative_l2_error"]),
    "context": ("context_tokens", ["latency_mean_ms", "kv_bytes_read"]),
}


def run_experiment(args) -> None:
    device = experiment_device(args.device)
    kwargs = {
        "values": args.values,
        "device": device,
        "precision": args.precision,
        "warmup": args.warmup,
        "repeats": args.repeats,
    }
    if args.name in {"batch", "depth", "precision"}:
        kwargs["width"] = args.width
    rows = MODULES[args.name].run(**kwargs)
    directory = Path(args.output_dir)
    csv_path = save_rows(rows, directory / "results" / f"{args.name}.csv")
    x, metrics = PLOTS[args.name]
    chart_path = plot_metrics(
        rows, x, metrics, directory / "charts" / f"{args.name}.png",
        f"{args.name.title()} scaling experiment",
    )
    print(f"\n{args.name.title()} experiment ({device})")
    for row in rows:
        print(row)
    print(f"\nSaved results: {csv_path}")
    print(f"Saved chart: {chart_path}")

