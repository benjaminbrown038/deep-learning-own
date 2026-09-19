#!/usr/bin/env bash
set -euo pipefail

mkdir -p results/nsight
nsys profile \
  --trace=cuda,nvtx,osrt \
  --sample=none \
  --force-overwrite=true \
  --output=results/nsight/qwen_fp16 \
  llm-lab benchmark \
    --size 1.5B \
    --precision fp16 \
    --context-tokens 512 \
    --tokens 50

