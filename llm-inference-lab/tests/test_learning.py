from llm_lab.learning.arithmetic import dot_product, matrix_vector_shape
from llm_lab.learning.memory import linear_memory
from llm_lab.learning.transformer import TransformerShape, decoder_token_cost


def test_dot_product_exposes_arithmetic():
    result = dot_product([1, 2], [3, 4], bias=1)
    assert result["output"] == 12
    assert result["flops"] == 4


def test_quantization_reduces_ideal_weight_bytes():
    fp16 = linear_memory(64, 64, "fp16")
    int4 = linear_memory(64, 64, "int4")
    assert int4["weight_bytes"] == fp16["weight_bytes"] / 4


def test_context_increases_kv_memory_linearly():
    shape = TransformerShape(2, 64, 4, 2, 16, 128, 1000)
    short = decoder_token_cost(shape, 128)
    long = decoder_token_cost(shape, 256)
    assert long["kv_cache_bytes"] == 2 * short["kv_cache_bytes"]


def test_batching_increases_matrix_flops_linearly():
    one = matrix_vector_shape(64, 64, 1)
    four = matrix_vector_shape(64, 64, 4)
    assert four["flops"] == 4 * one["flops"]

