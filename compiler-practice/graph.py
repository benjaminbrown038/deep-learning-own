"""A tiny graph and scheduler for one FP32 dense layer followed by ReLU."""

from dataclasses import asdict, dataclass
from typing import Tuple


@dataclass(frozen=True)
class Node:
    name: str
    operation: str
    inputs: Tuple[str, ...]
    shape: Tuple[int, ...]


@dataclass(frozen=True)
class Kernel:
    name: str
    operations: Tuple[str, ...]
    inputs: Tuple[str, ...]
    output: str


def build_graph(width: int, batch: int) -> Tuple[Node, ...]:
    """Frontend: represent Y = max(X @ W.T + bias, 0), without calculating it."""
    if width < 1 or batch < 1:
        raise ValueError("width and batch must be positive")
    return (
        Node("X", "input", (), (batch, width)),
        Node("W", "input", (), (width, width)),
        Node("bias", "input", (), (width,)),
        Node("Z", "matmul_transpose", ("X", "W"), (batch, width)),
        Node("A", "add_bias", ("Z", "bias"), (batch, width)),
        Node("Y", "relu", ("A",), (batch, width)),
    )


def schedule(graph: Tuple[Node, ...], mode: str) -> Tuple[Kernel, ...]:
    """One explicit fusion rule. This is deliberately not a general compiler."""
    if mode not in ("separate", "fused"):
        raise ValueError("mode must be separate or fused")
    if len(graph) != 6 or len(graph[0].shape) != 2:
        raise ValueError("only the dense + bias + ReLU graph is supported")
    batch, width = graph[0].shape
    if graph != build_graph(width, batch):
        raise ValueError("only the dense + bias + ReLU graph is supported")
    if mode == "fused":
        return (Kernel("dense_relu", ("Z", "A", "Y"), ("X", "W", "bias"), "Y"),)
    return tuple(Kernel(node.operation, (node.name,), node.inputs, node.name)
                 for node in graph if node.operation != "input")


def cost_model(graph: Tuple[Node, ...], kernels: Tuple[Kernel, ...]) -> dict:
    """Tensor-boundary accounting, not a prediction of DRAM traffic or time."""
    sizes = {}
    for node in graph:
        count = 1
        for dimension in node.shape:
            count *= dimension
        sizes[node.name] = count
    batch, width = graph[0].shape
    materialized = {n.name for n in graph if n.operation == "input"}
    materialized.update(k.output for k in kernels)
    boundary_values = sum(sum(sizes[name] for name in k.inputs) + sizes[k.output]
                          for k in kernels)
    return {
        "weight_values": width * width,
        "bias_values": width,
        "parameter_values_untrained": width * width + width,
        "matmul_flops_approx": 2 * batch * width * width,
        "bias_additions": batch * width,
        "relu_comparisons": batch * width,
        "kernel_calls_per_evaluation": len(kernels),
        "logical_tensor_storage_bytes": 4 * sum(sizes[n] for n in materialized),
        "temporary_tensor_bytes": 4 * sum(sizes[n] for n in materialized
                                          if n not in ("X", "W", "bias", "Y")),
        "ideal_tensor_boundary_bytes": 4 * boundary_values,
    }


def describe(graph: Tuple[Node, ...], kernels: Tuple[Kernel, ...]) -> dict:
    return {"nodes": [asdict(n) for n in graph],
            "kernels": [asdict(k) for k in kernels],
            "costs": cost_model(graph, kernels)}
