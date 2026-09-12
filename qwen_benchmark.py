"""Benchmark Qwen2.5 models from a terminal.

Examples:
    python qwen_benchmark.py --size 0.5B
    python qwen_benchmark.py --size 1.5B --tokens 50
"""

from __future__ import annotations

import argparse
import gc
import time

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


MODELS = {
    "0.5B": "Qwen/Qwen2.5-0.5B-Instruct",
    "1.5B": "Qwen/Qwen2.5-1.5B-Instruct",
    "3B": "Qwen/Qwen2.5-3B-Instruct",
    "7B": "Qwen/Qwen2.5-7B-Instruct",
    "14B": "Qwen/Qwen2.5-14B-Instruct",
}


# -----------------------------------------------------------------------------
# 1. Hardware
# -----------------------------------------------------------------------------

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


# -----------------------------------------------------------------------------
# 2. Architecture inspection (downloads configuration, not model weights)
# -----------------------------------------------------------------------------

def inspect_model(model_name: str) -> dict:
    config = AutoConfig.from_pretrained(model_name)
    width = config.hidden_size
    layers = config.num_hidden_layers

    return {
        "layers": layers,
        "hidden_width": width,
        "attention_heads": config.num_attention_heads,
        "kv_heads": config.num_key_value_heads,
        "head_dimension": width // config.num_attention_heads,
        "mlp_width": config.intermediate_size,
        "vocabulary": config.vocab_size,
        "rough_complexity_LxD2": layers * width**2,
    }


def print_architecture(size: str, architecture: dict) -> None:
    print(f"\n{size} architecture")
    print("-" * 32)
    for name, value in architecture.items():
        print(f"{name.replace('_', ' ').title():24} {value:,}")


# -----------------------------------------------------------------------------
# 3. Model loading
# -----------------------------------------------------------------------------

def load_model(model_name: str, device: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float16,
            device_map="auto",
        )
        input_device = next(model.parameters()).device
    elif device == "mps":
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float16,
            low_cpu_mem_usage=True,
        ).to("mps")
        input_device = torch.device("mps")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float32,
            low_cpu_mem_usage=True,
        )
        input_device = torch.device("cpu")

    return tokenizer, model.eval(), input_device


# -----------------------------------------------------------------------------
# 4. Benchmark
# -----------------------------------------------------------------------------

def build_context_prompt(
    tokenizer,
    base_prompt: str,
    target_tokens: int,
) -> str:
    """Repeat and trim text to create a controlled prompt-token length."""
    if target_tokens < 1:
        raise ValueError("context_tokens must be at least 1")

    base_ids = tokenizer(
        base_prompt + " ",
        add_special_tokens=False,
    )["input_ids"]

    repetitions = (target_tokens // len(base_ids)) + 1
    token_ids = (base_ids * repetitions)[:target_tokens]
    return tokenizer.decode(token_ids, skip_special_tokens=True)


def benchmark_model(
    model_name: str,
    prompt: str,
    max_new_tokens: int,
    context_tokens: int,
) -> dict:
    device = accelerator()
    print(f"\nLoading: {model_name}")
    print(f"Accelerator: {device}")

    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    tokenizer, model, input_device = load_model(model_name, device)
    prompt = build_context_prompt(tokenizer, prompt, context_tokens)

    messages = [{"role": "user", "content": prompt}]
    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(input_device)
    input_tokens = inputs["input_ids"].shape[1]

    # Warm up one short decode so first-run setup is excluded from the timer.
    with torch.inference_mode():
        model.generate(
            **inputs,
            max_new_tokens=3,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    synchronize(device)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    start = time.perf_counter()
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    synchronize(device)
    elapsed = time.perf_counter() - start

    generated_tokens = outputs.shape[1] - input_tokens
    parameter_count = sum(parameter.numel() for parameter in model.parameters())

    if device == "cuda":
        memory_gb = torch.cuda.max_memory_allocated() / 1e9
    elif device == "mps":
        memory_gb = torch.mps.current_allocated_memory() / 1e9
    else:
        memory_gb = None

    response = tokenizer.decode(
        outputs[0, input_tokens:],
        skip_special_tokens=True,
    )

    result = {
        "parameters": parameter_count,
        "requested_context_tokens": context_tokens,
        "input_tokens": input_tokens,
        "generated_tokens": generated_tokens,
        "seconds": elapsed,
        "tokens_per_second": generated_tokens / elapsed,
        "accelerator_memory_gb": memory_gb,
    }

    print("\nResults")
    print(f"Parameters: {parameter_count:,}")
    print(f"Generated tokens: {generated_tokens}")
    print(f"Time: {elapsed:.2f} seconds")
    print(f"Tokens/sec: {result['tokens_per_second']:.2f}")
    if memory_gb is not None:
        print(f"Accelerator memory: {memory_gb:.2f} GB")
    print(f"\nResponse:\n{response}")

    del outputs, inputs, model, tokenizer
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()

    return result


# -----------------------------------------------------------------------------
# 5. Command-line interface
# -----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark one Qwen2.5 model.")
    parser.add_argument("--size", choices=MODELS, default="0.5B")
    parser.add_argument("--tokens", type=int, default=100)
    parser.add_argument(
        "--context-tokens",
        type=int,
        default=128,
        help="Number of tokens placed in the user prompt.",
    )
    parser.add_argument(
        "--prompt",
        default="Explain how a CNC machine works.",
    )
    parser.add_argument(
        "--inspect-only",
        action="store_true",
        help="Print model architecture without downloading model weights.",
    )
    args = parser.parse_args()

    model_name = MODELS[args.size]
    print_architecture(args.size, inspect_model(model_name))

    if args.inspect_only:
        return

    benchmark_model(
        model_name,
        args.prompt,
        args.tokens,
        args.context_tokens,
    )


if __name__ == "__main__":
    main()