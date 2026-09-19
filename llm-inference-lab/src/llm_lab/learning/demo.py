from __future__ import annotations

from pprint import pprint

from .arithmetic import dot_product, matrix_vector_shape
from .memory import ideal_time_seconds, linear_memory
from .scaling import batch_comparison, precision_comparison
from .transformer import TransformerShape, decoder_token_cost


def run_learning_track(stage: str = "all") -> None:
    if stage in {"arithmetic", "all"}:
        print("\n1. Small neural-network calculation")
        pprint(dot_product([1.0, 2.0, -1.0], [0.5, -0.25, 2.0], bias=0.1))
        pprint(matrix_vector_shape(rows=32, columns=32))

    if stage in {"memory", "all"}:
        print("\n2. Memory tracking")
        operation = linear_memory(rows=4096, columns=4096, precision="fp16")
        pprint(operation)
        pprint(ideal_time_seconds(
            operation["flops"], operation["total_bytes"],
            compute_flops=65e12, bandwidth_bps=320e9,
        ))

    if stage in {"scaling", "all"}:
        print("\n3. Quantization")
        pprint(precision_comparison(rows=4096, columns=4096))
        print("\n3. Batching")
        pprint(batch_comparison(rows=4096, columns=4096, precision="fp16"))

    if stage in {"transformer", "all"}:
        print("\n4. Transformer layer and KV cache")
        shape = TransformerShape(
            layers=28,
            hidden_width=1536,
            attention_heads=12,
            kv_heads=2,
            head_dimension=128,
            mlp_width=8960,
            vocabulary=151936,
        )
        for context in (128, 512, 1024, 2048):
            pprint(decoder_token_cost(shape, context))

