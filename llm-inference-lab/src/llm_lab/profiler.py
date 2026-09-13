from __future__ import annotations

from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile, record_function


def profile_model(model, inputs, output_dir: str, device: str) -> str:
    activities = [ProfilerActivity.CPU]
    if device == "cuda":
        activities.append(ProfilerActivity.CUDA)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(str(destination)),
    ) as prof:
        with torch.inference_mode(), record_function("prefill"):
            outputs = model(**inputs, use_cache=True, return_dict=True)
        token = outputs.logits[:, -1:].argmax(dim=-1)
        with torch.inference_mode(), record_function("decode_one_token"):
            model(input_ids=token, past_key_values=outputs.past_key_values, use_cache=True)

    sort_key = "self_cuda_time_total" if device == "cuda" else "self_cpu_time_total"
    return prof.key_averages().table(sort_by=sort_key, row_limit=30)

