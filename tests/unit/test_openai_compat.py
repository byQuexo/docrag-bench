"""OpenAICompatProvider against a mocked HTTP server (httpx.MockTransport)."""

import json
from typing import Any

import httpx
import pytest

from docrag.answer.schema import Answer
from docrag.llm.openai_compat import OpenAICompatProvider, strip_think
from docrag.llm.types import Message, ProviderError


def _ok(content: str, **message_extra: Any) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "model": "m",
            "choices": [
                {
                    "message": {"content": content, **message_extra},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 12, "completion_tokens": 3},
        },
    )


# LEARN: httpx.MockTransport routes every request to a Python function instead of
# the network - a fake server in-process. Compare: MockWebServer (OkHttp) in
# Java, msw/nock in JS. No sockets, no flakiness, and we can inspect requests.
async def test_request_body_and_response_mapping() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return _ok('{"answer": "x", "evidence": []}')

    provider = OpenAICompatProvider(
        "test", "http://llm.local/v1", "secret", transport=httpx.MockTransport(handler)
    )
    messages = [
        Message(role="system", content="sys"),
        Message(role="user", content="q"),
    ]
    completion = await provider.chat("m", messages, Answer)
    await provider.aclose()

    request = seen[0]
    assert str(request.url) == "http://llm.local/v1/chat/completions"
    assert request.headers["Authorization"] == "Bearer secret"
    body = json.loads(request.content)
    assert body["messages"] == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "q"},
    ]
    rf = body["response_format"]
    assert rf["type"] == "json_schema"
    assert rf["json_schema"]["name"] == "Answer"
    assert rf["json_schema"]["strict"] is True
    assert rf["json_schema"]["schema"]["additionalProperties"] is False

    assert completion.content == '{"answer": "x", "evidence": []}'
    assert (completion.usage.input_tokens, completion.usage.output_tokens) == (12, 3)


async def test_retries_429_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    # Don't actually wait for the exponential backoff in a unit test.
    monkeypatch.setattr("docrag.llm.openai_compat.asyncio.sleep", fake_sleep)
    responses = iter([httpx.Response(429, text="slow down"), _ok("fine")])
    provider = OpenAICompatProvider(
        "t", "http://x/v1", transport=httpx.MockTransport(lambda r: next(responses))
    )
    completion = await provider.chat("m", [Message(role="user", content="q")])
    assert completion.content == "fine"
    assert sleeps == [1]


async def test_non_retryable_error_raises() -> None:
    provider = OpenAICompatProvider(
        "t",
        "http://x/v1",
        transport=httpx.MockTransport(lambda r: httpx.Response(400, text="bad")),
    )
    with pytest.raises(ProviderError, match="HTTP 400"):
        await provider.chat("m", [Message(role="user", content="q")])


def test_from_native_drops_reasoning_and_think_tags() -> None:
    provider = OpenAICompatProvider("t", "http://x/v1")
    raw = _ok("<think>secret plan</think>\nAnswer", reasoning_content="hidden").json()
    completion = provider.from_native("m", raw)
    assert completion.content == "Answer"
    assert "secret" not in completion.content and "hidden" not in completion.content


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("<think>a\nb</think>{}", "{}"),
        ("reasoning without opening tag</think> {}", "{}"),
        ("<THINK>x</THINK>ok", "ok"),
        ("plain", "plain"),
    ],
)
def test_strip_think(raw: str, expected: str) -> None:
    assert strip_think(raw) == expected


def test_max_tokens_is_always_sent() -> None:
    """Regression: without max_tokens OpenRouter reserved credit for 65,536 output
    tokens per call and rejected requests with HTTP 402."""
    from docrag.llm.openai_compat import OpenAICompatProvider
    from docrag.llm.types import Message

    provider = OpenAICompatProvider("x", "http://example.invalid/v1", max_tokens=512)
    body = provider.to_native("m", [Message(role="user", content="hi")], None, None)
    assert body["max_tokens"] == 512
    assert "temperature" not in body  # None = don't send
