"""Practice changing just width or batch. Uses the same timing as benchmark.py."""
from __future__ import annotations

import argparse
import random

from benchmark import add_settings, positive_int, print_result, run_case, save_results


def main():
    parser = argparse.ArgumentParser(description="Change one dimension; keep the benchmark method fixed.")
    parser.add_argument("variable", choices=["width", "batch"])
    parser.add_argument("--values", type=positive_int, nargs="+")
    add_settings(parser)
    args = parser.parse_args()
    values = args.values or ([64, 128, 256, 512, 1024] if args.variable == "width" else [1, 2, 4, 8, 16])
    if len(values) != len(set(values)):
        parser.error("--values must not contain duplicates")
    # Avoid always running the largest case last; the seed makes this reproducible.
    ordered_values = list(values)
    random.Random(args.seed).shuffle(ordered_values)
    print(f"Testing {args.variable} in this order: {ordered_values}")
    results = []
    try:
        for order, value in enumerate(ordered_values, start=1):
            case = argparse.Namespace(**vars(args))
            setattr(case, args.variable, value)
            result = run_case(case)
            result["sweep_variable"] = args.variable
            result["execution_order"] = order
            results.append(result)
            print_result(result)
    except (RuntimeError, ValueError, AssertionError) as error:
        if results:
            save_results(results, args.output, f"{args.variable}_partial")
        parser.exit(1, f"Sweep stopped: {error}\n")
    results.sort(key=lambda result: result[args.variable])
    save_results(results, args.output, args.variable)


if __name__ == "__main__":
    main()
