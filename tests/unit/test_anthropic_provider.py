"""AnthropicProvider format mapping (no network: we test to_native/from_native)."""

from types import SimpleNamespace

import pytest

from docrag.answer.schema import Answer
from docrag.llm.anthropic_provider import AnthropicProvider
from docrag.llm.types import Message, ProviderError


@pytest.fixture
def provider() -> AnthropicProvider:
    return AnthropicProvider(api_key="test-key", max_tokens=1000)


def test_to_native_moves_system_and_sets_output_config(
    provider: AnthropicProvider,
) -> None:
    native = provider.to_native(
        "claude-haiku-5-5",
        [
            Message(role="system", content="rules"),
            Message(role="system", content="context"),
            Message(role="user", content="q"),
            Message(role="assistant", content="a"),
            Message(role="user", content="q2"),
        ],
        Answer,
        0.0,
    )
    assert native["system"] == "rules\n\ncontext"
    assert [m["role"] for m in native["messages"]] == ["user", "assistant", "user"]
    assert native["max_tokens"] == 1000
    fmt = native["output_config"]["format"]
    assert fmt["type"] == "json_schema"
    assert fmt["schema"]["title"] == "Answer"
    assert "temperature" not in native


def test_to_native_without_schema(provider: AnthropicProvider) -> None:
    native = provider.to_native("m", [Message(role="user", content="q")], None, 0.0)
    assert "output_config" not in native and "system" not in native


def _response(stop_reason: str, *blocks: SimpleNamespace) -> SimpleNamespace:
    return SimpleNamespace(
        model="claude-haiku-5-5",
        stop_reason=stop_reason,
        content=list(blocks),
        usage=SimpleNamespace(input_tokens=50, output_tokens=7),
    )


def test_from_native_keeps_only_text_blocks(provider: AnthropicProvider) -> None:
    raw = _response(
        "end_turn",
        SimpleNamespace(type="thinking", thinking="internal"),
        SimpleNamespace(type="text", text='{"answer": '),
        SimpleNamespace(type="text", text='"x", "evidence": []}'),
    )
    completion = provider.from_native("m", raw)
    assert completion.content == '{"answer": "x", "evidence": []}'
    assert completion.usage.input_tokens == 50
    assert completion.finish_reason == "end_turn"


def test_refusal_raises(provider: AnthropicProvider) -> None:
    with pytest.raises(ProviderError, match="refused"):
        provider.from_native("m", _response("refusal"))
