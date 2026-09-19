from __future__ import annotations

from .memory import linear_memory


def precision_comparison(rows: int, columns: int, batch_size: int = 1) -> list[dict]:
    return [
        linear_memory(rows, columns, precision, batch_size)
        for precision in ("fp32", "fp16", "int8", "int4")
    ]


def batch_comparison(rows: int, columns: int, precision: str, batches=(1, 2, 4, 8)) -> list[dict]:
    results = []
    for batch in batches:
        result = linear_memory(rows, columns, precision, batch)
        result["batch_size"] = batch
        result["weight_bytes_per_output"] = result["weight_bytes"] / batch
        results.append(result)
    return results

