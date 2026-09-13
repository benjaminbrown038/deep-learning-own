from __future__ import annotations

import argparse
import gc

import torch

from .benchmark import benchmark_loaded_model, print_result
from .config import MODELS, PRECISIONS
from .hardware import accelerator, collect_hardware
from .model_loader import load_model
from .plotting import create_plots
from .profiler import profile_model
from .prompt import build_context_prompt, tokenize_chat
from .roofline import build_roofline_report, print_analysis
from .storage import append_result, save_run_json


def cleanup(device: str) -> None:
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()


def run_benchmark(args) -> None:
    device = accelerator()
    print("Hardware:", collect_hardware(device))
    contexts = args.context_sweep or [args.context_tokens]
    precisions = args.precision_sweep or [args.precision]

    for precision in precisions:
        print(f"\nLoading {MODELS[args.size]} ({precision}) on {device}")
        tokenizer, model, input_device = load_model(MODELS[args.size], device, precision)
        for context in contexts:
            print("\n" + "=" * 60)
            print(f"{args.size} / {precision} / context {context}")
            result = benchmark_loaded_model(
                tokenizer=tokenizer, model=model, input_device=input_device,
                model_name=MODELS[args.size], size=args.size, precision=precision,
                prompt=args.prompt, context_tokens=context,
                new_tokens=args.tokens, device=device,
            )
            print_result(result)
            append_result(result, args.results)
            save_run_json(result, args.run_directory)
        del tokenizer, model
        cleanup(device)


def run_profile(args) -> None:
    device = accelerator()
    tokenizer, model, input_device = load_model(MODELS[args.size], device, args.precision)
    prompt = build_context_prompt(tokenizer, args.prompt, args.context_tokens)
    inputs = tokenize_chat(tokenizer, prompt, input_device)
    print(profile_model(model, inputs, args.output, device))


def run_roofline(args) -> None:
    operations = build_roofline_report(
        model_name=MODELS[args.size],
        precision=args.precision,
        context_tokens=args.context_tokens,
        peak_tflops=args.peak_tflops,
        bandwidth_gbps=args.bandwidth_gbps,
        kernel_overhead_us=args.kernel_overhead_us,
        csv_path=args.csv,
        plot_path=args.plot,
    )
    print_analysis(operations)
    print(f"\nSaved table: {args.csv}")
    print(f"Saved chart: {args.plot}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LLM inference performance laboratory")
    sub = parser.add_subparsers(dest="command", required=True)

    benchmark = sub.add_parser("benchmark", help="Run one benchmark or a sweep")
    benchmark.add_argument("--size", choices=MODELS, default="0.5B")
    benchmark.add_argument("--precision", choices=PRECISIONS, default="fp16")
    benchmark.add_argument("--precision-sweep", nargs="+", choices=PRECISIONS)
    benchmark.add_argument("--context-tokens", type=int, default=128)
    benchmark.add_argument("--context-sweep", type=int, nargs="+")
    benchmark.add_argument("--tokens", type=int, default=50)
    benchmark.add_argument("--prompt", default="Explain how a CNC machine works.")
    benchmark.add_argument("--results", default="results/benchmarks.csv")
    benchmark.add_argument("--run-directory", default="results/runs")
    benchmark.set_defaults(func=run_benchmark)

    plot = sub.add_parser("plot", help="Create comparison charts from CSV results")
    plot.add_argument("--results", default="results/benchmarks.csv")
    plot.add_argument("--output", default="charts")
    plot.set_defaults(func=lambda args: [print(path) for path in create_plots(args.results, args.output)])

    profiling = sub.add_parser("profile", help="Capture a PyTorch profiler trace")
    profiling.add_argument("--size", choices=MODELS, default="0.5B")
    profiling.add_argument("--precision", choices=PRECISIONS, default="fp16")
    profiling.add_argument("--context-tokens", type=int, default=128)
    profiling.add_argument("--prompt", default="Explain how a CNC machine works.")
    profiling.add_argument("--output", default="results/profiler")
    profiling.set_defaults(func=run_profile)

    roofline = sub.add_parser(
        "roofline",
        help="Estimate operation-level compute, memory, and launch bottlenecks",
    )
    roofline.add_argument("--size", choices=MODELS, default="1.5B")
    roofline.add_argument("--precision", choices=PRECISIONS, default="fp16")
    roofline.add_argument("--context-tokens", type=int, default=512)
    roofline.add_argument("--peak-tflops", type=float, required=True)
    roofline.add_argument("--bandwidth-gbps", type=float, required=True)
    roofline.add_argument("--kernel-overhead-us", type=float, default=5.0)
    roofline.add_argument("--csv", default="results/roofline.csv")
    roofline.add_argument("--plot", default="charts/roofline.png")
    roofline.set_defaults(func=run_roofline)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
