"""Any OpenAI-compatible /chat/completions endpoint.

One class covers OpenAI, Groq, Mistral, DeepSeek, Gemini (OpenAI-compat endpoint)
and local servers (Ollama, LM Studio, vLLM, llama.cpp). Only base_url + key differ.

We use plain httpx instead of the `openai` SDK on purpose: the conversion is
the point of this module, and httpx.MockTransport makes it trivially testable.
"""

import asyncio
import re
from typing import Any

import httpx
from pydantic import BaseModel

from docrag.llm.base import Provider
from docrag.llm.types import (
    ChatCompletion,
    ImagePart,
    Message,
    ProviderError,
    Usage,
    json_schema_for,
)

_RETRY_STATUS = {408, 409, 429, 500, 502, 503, 504}


class OpenAICompatProvider(Provider):
    def __init__(
        self,
        name: str,
        base_url: str,
        api_key: str | None = None,
        *,
        timeout: float = 120.0,
        max_retries: int = 2,
        transport: httpx.AsyncBaseTransport | None = None,  # tests inject a mock here
        extra_body: dict[str, Any] | None = None,
        max_tokens: int = 4096,
    ) -> None:
        self.name = name
        # Provider-specific top-level fields merged into every request body
        # (e.g. OpenRouter's routing options). Keeps the class generic.
        self._extra_body = extra_body or {}
        # Always cap the output. Without it OpenRouter reserved credit for the
        # model's maximum (65,536 tokens) per request and rejected calls with
        # HTTP 402 once the balance was lower than that - in our first full run.
        self._max_tokens = max_tokens
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/") + "/",
            headers=headers,
            timeout=timeout,
            transport=transport,
        )
        self._max_retries = max_retries

    def to_native(
        self,
        model: str,
        messages: list[Message],
        response_format: type[BaseModel] | None,
        temperature: float | None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": model,
            "messages": [_to_openai_message(m) for m in messages],
        }
        # None = don't send it at all. Some models (e.g. Claude 5.5 via OpenRouter)
        # reject any sampling parameter, so "0.0" is not a safe universal default.
        body["max_tokens"] = self._max_tokens
        if temperature is not None:
            body["temperature"] = temperature
        if response_format is not None:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": response_format.__name__,
                    "schema": json_schema_for(response_format),
                    "strict": True,
                },
            }
        return {**body, **self._extra_body}

    async def _call(self, native: dict[str, Any]) -> Any:
        for attempt in range(self._max_retries + 1):
            try:
                resp = await self._client.post("chat/completions", json=native)
            except httpx.TransportError as err:  # connection refused, timeout, ...
                if attempt == self._max_retries:
                    raise ProviderError(f"{self.name}: {err!r}") from err
            else:
                if resp.status_code < 400:
                    return resp.json()
                if (
                    resp.status_code not in _RETRY_STATUS
                    or attempt == self._max_retries
                ):
                    raise ProviderError(
                        f"{self.name}: HTTP {resp.status_code}: {resp.text[:500]}"
                    )
            # Exponential backoff: 1s, 2s, 4s ... (rate limits recover with time).
            await asyncio.sleep(2**attempt)
        raise AssertionError("unreachable")

    def from_native(self, model: str, raw: Any) -> ChatCompletion:
        choice = raw["choices"][0]
        message = choice["message"]
        # Some servers return reasoning separately ("reasoning_content"/"reasoning").
        # We drop it on purpose: reasoning is never treated as answer text.
        content = strip_think(message.get("content") or "")
        usage = raw.get("usage") or {}
        return ChatCompletion(
            model=raw.get("model", model),
            content=content,
            usage=Usage(
                input_tokens=usage.get("prompt_tokens", 0),
                output_tokens=usage.get("completion_tokens", 0),
                # OpenRouter reports the actually billed price (USD) per call.
                cost_usd=usage.get("cost"),
            ),
            finish_reason=choice.get("finish_reason"),
        )

    async def aclose(self) -> None:
        await self._client.aclose()


_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def strip_think(text: str) -> str:
    """Remove <think>...</think> blocks that local reasoning models put in content.

    Must happen BEFORE JSON validation, or every answer of e.g. Qwen3/DeepSeek-R1
    fails to parse.
    """
    text = _THINK.sub("", text)
    # Some templates drop the opening tag and only emit "...</think>answer".
    if "</think>" in text.lower():
        text = re.split(r"</think>", text, flags=re.IGNORECASE)[-1]
    return text.strip()


def _to_openai_message(m: Message) -> dict[str, Any]:
    """Our message -> OpenAI chat format. Images become `image_url` data URLs."""
    if isinstance(m.content, str):
        return {"role": m.role, "content": m.content}
    parts: list[dict[str, Any]] = []
    for part in m.content:
        if isinstance(part, ImagePart):
            url = f"data:{part.media_type};base64,{part.data}"
            parts.append({"type": "image_url", "image_url": {"url": url}})
        else:
            parts.append({"type": "text", "text": part.text})
    return {"role": m.role, "content": parts}
