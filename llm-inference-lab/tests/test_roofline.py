from types import SimpleNamespace

from llm_lab.roofline import analyze_operations, decoder_operations


CONFIG = SimpleNamespace(
    num_hidden_layers=2,
    hidden_size=64,
    num_attention_heads=4,
    num_key_value_heads=2,
    intermediate_size=128,
    vocab_size=1000,
)


def test_decoder_estimates_are_positive():
    operations = decoder_operations(CONFIG, context_tokens=128, precision="fp16")
    assert all(op.flops > 0 and op.bytes_moved > 0 for op in operations)


def test_analysis_assigns_bottlenecks_and_shares():
    operations = decoder_operations(CONFIG, context_tokens=128, precision="fp16")
    analyze_operations(operations, peak_tflops=10, bandwidth_gbps=300, kernel_overhead_us=1)
    assert all(op.bottleneck in {"compute", "memory bandwidth", "kernel launch overhead"} for op in operations)
    assert abs(sum(op.predicted_share_percent for op in operations) - 100) < 1e-9

