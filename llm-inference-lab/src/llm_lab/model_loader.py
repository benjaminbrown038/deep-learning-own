from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def quantization_config(precision: str):
    if precision == "int8":
        return BitsAndBytesConfig(load_in_8bit=True)
    if precision == "int4":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
    return None


def load_model(model_name: str, device: str, precision: str):
    if precision in {"int8", "int4"} and device != "cuda":
        raise ValueError("INT8/INT4 bitsandbytes tests require an NVIDIA CUDA GPU")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    qconfig = quantization_config(precision)

    if precision == "fp32":
        dtype = torch.float32
    elif precision == "bf16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float16

    kwargs = {"torch_dtype": dtype, "low_cpu_mem_usage": True}
    if device == "cuda":
        kwargs["device_map"] = "auto"
        if qconfig is not None:
            kwargs["quantization_config"] = qconfig

    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    if device in {"mps", "cpu"}:
        model = model.to(device)
    input_device = next(model.parameters()).device
    return tokenizer, model.eval(), input_device

