"""Trace one equation through a small compiler and a real CPU runtime.

Run: python compiler_lab.py --width 32 --inspect-only
     python compiler_lab.py --width 32
"""

import argparse
import csv
import json
import platform
import random
import statistics
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

from codegen import emit_c
from graph import build_graph, cost_model, describe, schedule
from runtime import Executable, compile_source, find_compiler, make_inputs, validate_output


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def parser():
    result = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    result.add_argument("--width", type=positive_int, default=128)
    result.add_argument("--batch", type=positive_int, default=1)
    result.add_argument("--mode", choices=("both", "separate", "fused"), default="both")
    result.add_argument("--inspect-only", action="store_true",
                        help="Show graph, schedule and byte counts; do not compile or run.")
    result.add_argument("--show-code", action="store_true", help="Also print generated C.")
    result.add_argument("--emit-assembly", action="store_true", help="Save compiler assembly too.")
    result.add_argument("--warmup", type=positive_int, default=5)
    result.add_argument("--iterations", type=positive_int, default=200)
    result.add_argument("--repeats", type=positive_int, default=7)
    result.add_argument("--seed", type=int, default=0)
    result.add_argument("--compiler", help="Compiler executable name or path (no extra flags).")
    result.add_argument("--output", type=Path, default=Path("results"))
    return result


def print_inspection(graph, schedules):
    print("\n1. Equation: Y = max(X @ W.T + bias, 0)\n2. Computation graph")
    for node in graph:
        arguments = ", ".join(node.inputs)
        print(f"   {node.name:5} = {node.operation}({arguments})  shape={node.shape}")
    costs = cost_model(graph, next(iter(schedules.values())))
    print(f"   Untrained weight + bias values: {costs['parameter_values_untrained']:,}")
    print(f"   Matmul FLOPs: ~{costs['matmul_flops_approx']:,}; "
          f"bias additions: {costs['bias_additions']:,}; "
          f"ReLU comparisons: {costs['relu_comparisons']:,}")
    print("\n3. Kernel schedules (same equation)")
    for mode, kernels in schedules.items():
        for kernel in kernels:
            print(f"   {mode:8} | {kernel.name}({', '.join(kernel.inputs)}) -> {kernel.output}")
        costs = cost_model(graph, kernels)
        print(f"            Calls: {len(kernels)}; temporary tensors: "
              f"{costs['temporary_tensor_bytes']:,} bytes; "
              f"logical storage: {costs['logical_tensor_storage_bytes']:,} bytes")
        print(f"            Ideal tensor-boundary bytes: "
              f"{costs['ideal_tensor_boundary_bytes']:,} (not measured DRAM traffic)")


def run(args):
    graph = build_graph(args.width, args.batch)
    modes = ["separate", "fused"] if args.mode == "both" else [args.mode]
    schedules = {mode: schedule(graph, mode) for mode in modes}
    sources = {mode: emit_c(graph, kernels) for mode, kernels in schedules.items()}
    print_inspection(graph, schedules)
    if args.show_code:
        for mode, source in sources.items():
            print(f"\nGenerated {mode}.c:\n{source}")
    if args.inspect_only:
        return None

    compiler = find_compiler(args.compiler)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    destination = args.output / f"compiler_{stamp}_{uuid.uuid4().hex[:8]}"
    destination.mkdir(parents=True, exist_ok=False)
    graph_data = {mode: describe(graph, kernels) for mode, kernels in schedules.items()}
    (destination / "graph_and_schedule.json").write_text(json.dumps(graph_data, indent=2) + "\n")
    inputs = make_inputs(args.width, args.batch, args.seed)
    executables, builds, checks = {}, {}, {}
    print("\n4. Generate C, compile machine code, and load CPU kernels")
    for mode in modes:
        binary, builds[mode] = compile_source(sources[mode], destination, mode,
                                              compiler, args.emit_assembly)
        executables[mode] = Executable(graph, schedules[mode], binary, inputs)
        checks[mode] = validate_output(executables[mode], args.width, args.batch, args.seed)
        print(f"   {mode}: compiled in {builds[mode]['compile_seconds']:.3f} s; "
              f"output check passed ({checks[mode]['checked_outputs']}/"
              f"{checks[mode]['total_outputs']} values)")
    for executable in executables.values():
        for _ in range(args.warmup):
            executable.run()

    print("\n5. Measure warm execution (Python dispatch + synchronous CPU kernels)")
    samples = {mode: [] for mode in modes}
    initial_order = modes.copy()
    random.Random(args.seed).shuffle(initial_order)
    orders = []
    for repeat in range(args.repeats):
        order = initial_order if repeat % 2 == 0 else list(reversed(initial_order))
        orders.append(list(order))
        for mode in order:
            samples[mode].append(executables[mode].sample(args.iterations))

    rows = []
    for mode in modes:
        per_call = [elapsed / args.iterations for elapsed in samples[mode]]
        median = statistics.median(per_call)
        row = {"mode": mode, "width": args.width, "batch": args.batch,
               "dtype": "fp32", "device": "cpu", "warmup": args.warmup,
               "iterations": args.iterations, "repeats": args.repeats, "seed": args.seed,
               **cost_model(graph, schedules[mode]), **checks[mode],
               "compile_seconds": builds[mode]["compile_seconds"],
               "median_seconds_per_evaluation": median,
               "min_seconds_per_evaluation": min(per_call),
               "max_seconds_per_evaluation": max(per_call),
               "examples_per_second": args.batch / median}
        rows.append(row)
        print(f"   {mode:8}: median {median * 1e6:.3f} us/evaluation; "
              f"range {min(per_call) * 1e6:.3f}..{max(per_call) * 1e6:.3f}")
    if len(rows) == 2:
        ratio = rows[0]["median_seconds_per_evaluation"] / rows[1]["median_seconds_per_evaluation"]
        print(f"   Separate time / fused time = {ratio:.3f} (above 1 means fused was faster)")
    with (destination / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    details = {
        "equation": "Y = max(X @ W.T + bias, 0)",
        "timer_scope": "Python dispatch plus synchronous single-thread CPU C kernels",
        "excluded_from_timer": ["graph building", "compilation", "allocation", "validation", "saving"],
        "platform": platform.platform(), "machine": platform.machine(),
        "processor": platform.processor(), "python": sys.version,
        "cpu_model_note": "processor can be empty; hardware clock and RAM traffic are not measured",
        "sample_order": orders, "sample_total_seconds": samples,
        "builds": builds, "results": rows,
    }
    (destination / "details.json").write_text(json.dumps(details, indent=2) + "\n")
    print(f"\nSaved: {destination.resolve()}")
    print("Read the generated .c files to see what changed. This is not an LLM tokens/s test.")
    return destination


def main():
    args = parser().parse_args()
    try:
        run(args)
    except (RuntimeError, OSError, MemoryError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
