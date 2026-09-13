import pytest

from llm_lab.prompt import build_context_prompt


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        return text.split()

    def decode(self, tokens, skip_special_tokens=True):
        return " ".join(tokens)


def test_prompt_preserves_question():
    result = build_context_prompt(FakeTokenizer(), "What is CNC?", 20)
    assert "Question: What is CNC?" in result


def test_prompt_rejects_zero_target():
    with pytest.raises(ValueError):
        build_context_prompt(FakeTokenizer(), "test", 0)

