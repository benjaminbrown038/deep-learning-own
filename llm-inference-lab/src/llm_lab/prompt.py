from __future__ import annotations


FILLER = (
    "Computer numerical control machines automate manufacturing processes. "
    "Programmed instructions control tool position, spindle speed, feed rate, "
    "and cutting operations. "
)


def build_context_prompt(tokenizer, question: str, target_tokens: int) -> str:
    """Build an approximately fixed-length prompt while preserving the question."""
    if target_tokens < 1:
        raise ValueError("target_tokens must be at least 1")
    question_text = f"\n\nQuestion: {question}"
    filler_ids = tokenizer.encode(FILLER, add_special_tokens=False)
    question_ids = tokenizer.encode(question_text, add_special_tokens=False)
    filler_target = max(target_tokens - len(question_ids), 0)
    repeats = (filler_target // len(filler_ids)) + 1
    prompt_ids = (filler_ids * repeats)[:filler_target] + question_ids
    return tokenizer.decode(prompt_ids, skip_special_tokens=True)


def tokenize_chat(tokenizer, prompt: str, input_device):
    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return tokenizer(formatted, return_tensors="pt").to(input_device)

