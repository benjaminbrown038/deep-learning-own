from __future__ import annotations

import os
import platform
import sys

import psutil
import torch
import transformers


def accelerator() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def synchronize(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def collect_hardware(device: str | None = None) -> dict:
    device = device or accelerator()
    info = {
        "platform": platform.platform(),
        "python_version": sys.version.split()[0],
        "pytorch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "device": device,
        "cpu": platform.processor() or platform.machine() or "unknown",
        "cpu_count": os.cpu_count(),
        "system_memory_gb": round(psutil.virtual_memory().total / 1e9, 3),
        "cuda_version": torch.version.cuda,
        "accelerator_name": "CPU",
        "accelerator_total_memory_gb": None,
        "compute_capability": None,
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    if device == "cuda":
        props = torch.cuda.get_device_properties(0)
        info.update({
            "accelerator_name": props.name,
            "accelerator_total_memory_gb": round(props.total_memory / 1e9, 3),
            "compute_capability": f"{props.major}.{props.minor}",
        })
    elif device == "mps":
        info["accelerator_name"] = "Apple MPS"
        info["gpu_count"] = 1
    return info


def peak_memory_gb(device: str) -> float | None:
    if device == "cuda":
        return torch.cuda.max_memory_allocated() / 1e9
    if device == "mps":
        return torch.mps.current_allocated_memory() / 1e9
    return None


def reset_peak_memory(device: str) -> None:
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

