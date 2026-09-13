# LLM Inference Lab

A reproducible project for studying how model scale, context length, numerical
precision, hardware, and CUDA execution affect Qwen2.5 inference.

## What it implements

- exact prefill and manual token-by-token decode timing;
- KV-cache reuse during greedy decoding;
- context-length and precision sweeps;
- FP32, FP16, BF16, INT8, and INT4 loading;
- hardware and software metadata collection;
- per-run JSON plus cumulative CSV storage;
- latency mean, p50, p95, and p99;
- automatic performance and memory plots;
- PyTorch Profiler traces and kernel tables;
- an NVIDIA Nsight Systems launch script;
- historical results from the earlier Apple MPS and Tesla T4 experiments.

## Installation

Mac or CPU:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[test]"
```

NVIDIA/Colab, including quantization:

```bash
pip install -e ".[cuda,test]"
```

## Benchmark commands

One configuration:

```bash
llm-lab benchmark --size 1.5B --precision fp16 --context-tokens 512 --tokens 50
```

Context sweep, loading the model only once:

```bash
llm-lab benchmark \
  --size 1.5B \
  --precision fp16 \
  --context-sweep 128 512 1024 2048 \
  --tokens 50
```

Precision sweep on an NVIDIA GPU:

```bash
llm-lab benchmark \
  --size 1.5B \
  --precision-sweep fp16 int8 int4 \
  --context-tokens 512 \
  --tokens 50
```

Results are appended to `results/benchmarks.csv`; complete runs, including the
generated response and per-token latencies, are saved under `results/runs/`.

## Plot results

```bash
llm-lab plot --results results/benchmarks.csv --output charts
```

Plot the included historical data:

```bash
llm-lab plot --results results/historical_benchmarks.csv --output charts/historical
```

Historical throughput is stored in the decode column for chart compatibility,
but it represents the older combined `model.generate()` measurement. It is not
an exact decode-only measurement.

## PyTorch Profiler

```bash
llm-lab profile \
  --size 1.5B \
  --precision fp16 \
  --context-tokens 512 \
  --output results/profiler
```

Then inspect the trace:

```bash
tensorboard --logdir results/profiler
```

## NVIDIA Nsight Systems

On an NVIDIA Linux machine with `nsys` installed:

```bash
bash scripts/run_nsight.sh
```

Open the resulting `.nsys-rep` file in NVIDIA Nsight Systems.

## How to interpret bottlenecks

| Evidence | Likely bottleneck |
|---|---|
| GEMM/GEMV kernels dominate CUDA time | Weight projection mathematics or traffic |
| Attention kernels grow strongly with context | Attention/KV-cache access |
| Quantization conversion kernels dominate | Dequantization overhead |
| Many short kernels separated by gaps | Launch/runtime overhead |
| Low compute use with sustained memory traffic | Memory bandwidth |
| Copies and allocations dominate | Memory management |

The central performance model is:

```text
operation time ≈ max(operations / compute rate, bytes moved / memory bandwidth)
                 + launch and runtime overhead
```

## Current limitations

- Greedy decoding is used for deterministic timing, not sampled generation.
- INT8 and INT4 use bitsandbytes and require a supported NVIDIA CUDA setup.
- The first generated token is counted with prefill; decode statistics cover
  subsequent token iterations.
- Energy and quality evaluation are planned extensions.

