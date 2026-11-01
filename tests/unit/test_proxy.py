"""LLMProxy: routing, cache, think tags, structured-output retry, budget, order."""

import json

import pytest

from docrag.answer.schema import Answer
from docrag.llm.fake import FakeProvider
from docrag.llm.proxy import (
    BudgetExceededError,
    LLMProxy,
    StructuredOutputError,
)
from docrag.llm.types import Message
from docrag.shared.cache import Cache

USER = [Message(role="user", content="hi")]
VALID = json.dumps({"answer": "ok", "evidence": []})


def scripted(*replies: str) -> FakeProvider:
    """A FakeProvider that returns the given replies in order."""
    it = iter(replies)
    # LEARN: a lambda ignoring its arguments; next(it) pulls the next reply.
    return FakeProvider(responder=lambda messages, schema: next(it))


def test_resolve_splits_on_first_colon_only() -> None:
    ollama = FakeProvider()
    proxy = LLMProxy({"ollama": ollama})
    provider, model = proxy.resolve("ollama:qwen3:8b")
    assert provider is ollama
    assert model == "qwen3:8b"


@pytest.mark.parametrize("route", ["ollama", "ollama:", "missing:model"])
def test_resolve_errors(route: str) -> None:
    with pytest.raises(ValueError):
        LLMProxy({"ollama": FakeProvider()}).resolve(route)


async def test_cache_hit_skips_provider() -> None:
    provider = FakeProvider()
    proxy = LLMProxy({"fake": provider}, cache=Cache(":memory:"))
    first = await proxy.chat("fake:m", USER)
    second = await proxy.chat("fake:m", USER)
    assert first == second
    assert len(provider.calls) == 1  # second call served from the cache
    assert proxy.stats.cache_hits == 1
    # A different prompt is a different key.
    await proxy.chat("fake:m", [Message(role="user", content="other")])
    assert len(provider.calls) == 2


async def test_structured_parses_and_unwraps_fence() -> None:
    proxy = LLMProxy({"fake": scripted(f"```json\n{VALID}\n```")})
    parsed, _ = await proxy.structured("fake:m", USER, Answer)
    assert parsed == Answer(answer="ok", evidence=[])


async def test_one_retry_then_success() -> None:
    provider = scripted("not json", VALID)
    proxy = LLMProxy({"fake": provider})
    parsed, _ = await proxy.structured("fake:m", USER, Answer)
    assert parsed.answer == "ok"
    assert proxy.stats.validation_retries == 1
    # The retry shows the model its own bad reply plus the validation error.
    retry_messages = provider.calls[1]["messages"]
    assert retry_messages[-2].role == "assistant"
    assert retry_messages[-2].content == "not json"
    assert "did not match" in retry_messages[-1].content


async def test_second_failure_raises_and_is_counted() -> None:
    proxy = LLMProxy({"fake": scripted("bad", "still bad", VALID)})
    with pytest.raises(StructuredOutputError):
        await proxy.structured("fake:m", USER, Answer)
    assert proxy.stats.validation_failures == 1
    assert proxy.stats.calls == 2  # exactly ONE retry, never a third call


async def test_budget_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    # LEARN: monkeypatch replaces an attribute for the duration of ONE test and
    # restores it afterwards (like jest.spyOn(...).mockReturnValue or Mockito).
    # We patch the name where it is LOOKED UP (docrag.llm.proxy), not where
    # it is defined - the most common mocking mistake in Python.
    monkeypatch.setattr("docrag.llm.proxy.cost_eur", lambda route, usage: 0.6)
    proxy = LLMProxy({"fake": FakeProvider()}, budget_eur=1.0)
    await proxy.chat("fake:m", [Message(role="user", content="1")])
    await proxy.chat("fake:m", [Message(role="user", content="2")])  # 0.6 < 1.0
    assert proxy.stats.cost_eur == pytest.approx(1.2)
    with pytest.raises(BudgetExceededError):
        await proxy.chat("fake:m", [Message(role="user", content="3")])


@pytest.mark.parametrize(
    "messages",
    [
        [Message(role="user", content="a"), Message(role="system", content="late")],
        [Message(role="system", content="only system")],
    ],
)
async def test_message_order_is_checked(messages: list[Message]) -> None:
    with pytest.raises(ValueError):
        await LLMProxy({"fake": FakeProvider()}).chat("fake:m", messages)
