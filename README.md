# deep-learning-own

# Qwen2.5 Benchmark Project

This project measures how Qwen2.5 model size and context length affect:

- Generation speed
- Accelerator memory
- Tokens per second
- Model architecture

## Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the required packages:

```bash
python -m pip install torch transformers accelerate
```

## Inspect Model Architecture

Inspect the 0.5B model without downloading its weights:

```bash
python qwen_benchmark.py --size 0.5B --inspect-only
```

Inspect the 1.5B model:

```bash
python qwen_benchmark.py --size 1.5B --inspect-only
```

## Benchmark the 0.5B Model

### 128-token context

```bash
python qwen_benchmark.py \
  --size 0.5B \
  --context-tokens 128 \
  --tokens 50
```

### 512-token context

```bash
python qwen_benchmark.py \
  --size 0.5B \
  --context-tokens 512 \
  --tokens 50
```

### 1,024-token context

```bash
python qwen_benchmark.py \
  --size 0.5B \
  --context-tokens 1024 \
  --tokens 50
```

## Benchmark the 1.5B Model

### 128-token context

```bash
python qwen_benchmark.py \
  --size 1.5B \
  --context-tokens 128 \
  --tokens 50
```

### 512-token context

```bash
python qwen_benchmark.py \
  --size 1.5B \
  --context-tokens 512 \
  --tokens 50
```

### 1,024-token context

```bash
python qwen_benchmark.py \
  --size 1.5B \
  --context-tokens 1024 \
  --tokens 50
```

## Current 0.5B Results

| Model | Context | Generated tokens | Time | Tokens/sec | Memory |
|---|---:|---:|---:|---:|---:|
| Qwen2.5-0.5B | 128 | 50 | 1.22 s | 41.09 | 0.99 GB |
| Qwen2.5-0.5B | 512 | 50 | 1.31 s | 38.08 | 0.99 GB |
| Qwen2.5-0.5B | 1,024 | 50 | 1.22 s | 41.09 | 0.99 GB |

## Command Options

```text
--size             Model size: 0.5B, 1.5B, 3B, 7B, or 14B
--tokens           Number of new tokens to generate
--context-tokens   Number of tokens in the test prompt
--prompt           Base prompt used for the experiment
--inspect-only     Display architecture without loading model weights
```

## Experimental Method

Keep model settings constant and change one variable at a time:

\[
\text{model size}
\rightarrow
\text{context length}
\rightarrow
\text{memory}
\rightarrow
\text{tokens/sec}
\]

The next experiment scales the model from 0.5B to 1.5B while preserving the same context lengths and output length.