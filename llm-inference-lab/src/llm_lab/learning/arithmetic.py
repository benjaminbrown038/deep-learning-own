from __future__ import annotations


def dot_product(values: list[float], weights: list[float], bias: float = 0.0) -> dict:
    """Compute one neuron explicitly and return every multiply/add step."""
    if len(values) != len(weights):
        raise ValueError("values and weights must have the same length")
    products = [value * weight for value, weight in zip(values, weights)]
    output = sum(products) + bias
    return {
        "values": values,
        "weights": weights,
        "products": products,
        "bias": bias,
        "output": output,
        "multiplications": len(values),
        "additions": len(values),
        "flops": 2 * len(values),
    }


def matrix_vector_shape(rows: int, columns: int, batch_size: int = 1) -> dict:
    """Operation count for Y = XW at the shapes used by linear layers."""
    if min(rows, columns, batch_size) < 1:
        raise ValueError("all dimensions must be positive")
    return {
        "rows": rows,
        "columns": columns,
        "batch_size": batch_size,
        "output_values": batch_size * rows,
        "multiply_add_pairs": batch_size * rows * columns,
        "flops": 2 * batch_size * rows * columns,
    }

