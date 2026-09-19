from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class TransformerShape:
    layers: int
    hidden_width: int
    attention_heads: int
    kv_heads: int
    head_dimension: int
    mlp_width: int
    vocabulary: int


def decoder_token_cost(shape: TransformerShape, context: int, element_bytes: float = 2.0) -> dict:
    """Approximate arithmetic and memory for one new batch-one decode token."""
    if context < 1 or element_bytes <= 0:
        raise ValueError("context and element_bytes must be positive")
    L, D, F = shape.layers, shape.hidden_width, shape.mlp_width
    d, Hkv, V = shape.head_dimension, shape.kv_heads, shape.vocabulary
    projection_flops = L * (8 * D * D + 6 * D * F)
    attention_flops = 4 * L * context * D
    lm_head_flops = 2 * D * V
    kv_bytes_per_token = 2 * L * Hkv * d * element_bytes
    kv_cache_bytes = kv_bytes_per_token * context
    kv_read_bytes = kv_cache_bytes
    return {
        **asdict(shape),
        "context": context,
        "projection_flops": projection_flops,
        "attention_flops": attention_flops,
        "lm_head_flops": lm_head_flops,
        "total_flops_per_decode_token": projection_flops + attention_flops + lm_head_flops,
        "kv_bytes_added_per_token": kv_bytes_per_token,
        "kv_cache_bytes": kv_cache_bytes,
        "kv_bytes_read_per_decode_token": kv_read_bytes,
    }

