"""Anthropic Messages API adapter (official `anthropic` SDK).

Differences from the OpenAI format that this adapter maps:
* ``system`` is a top-level field, not a message with role "system".
* ``max_tokens`` is required.
* Structured output goes in ``output_config.format`` (JSON schema).
* The response is a list of content blocks; we keep only ``text`` blocks.
  ``thinking`` blocks are dropped - never shown as answer text.
* ``stop_reason == "refusal"`` is a 200 response that must be treated as an error.

(File is named anthropic_provider.py, not anthropic.py, so it doesn't shadow
the `anthropic` package on import - a classic Python gotcha.)
"""

from typing import Any

import anthropic
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


class AnthropicProvider(Provider):
    name = "anthropic"

    def __init__(self, api_key: str | None, max_tokens: int = 16000) -> None:
        # The SDK retries 429/5xx/connection errors itself (max_retries=2).
        self._client = anthropic.AsyncAnthropic(api_key=api_key)
        self._max_tokens = max_tokens

    def to_native(
        self,
        model: str,
        messages: list[Message],
        response_format: type[BaseModel] | None,
        temperature: float | None,
    ) -> dict[str, Any]:
        system = "\n\n".join(m.text for m in messages if m.role == "system")
        native: dict[str, Any] = {
            "model": model,
            "max_tokens": self._max_tokens,
            "messages": [
                {"role": m.role, "content": _to_anthropic_content(m)}
                for m in messages
                if m.role != "system"
            ],
        }
        if system:
            native["system"] = system
        if response_format is not None:
            native["output_config"] = {
                "format": {
                    "type": "json_schema",
                    "schema": json_schema_for(response_format),
                }
            }
        # Note: no `temperature` - current Claude models reject non-default sampling
        # parameters; output length/effort is the control knob instead.
        return native

    async def _call(self, native: dict[str, Any]) -> Any:
        try:
            return await self._client.messages.create(**native)
        except anthropic.APIStatusError as err:
            raise ProviderError(
                f"anthropic: HTTP {err.status_code}: {err.message}"
            ) from err
        except anthropic.APIConnectionError as err:
            raise ProviderError(f"anthropic: connection error: {err}") from err

    def from_native(self, model: str, raw: Any) -> ChatCompletion:
        if raw.stop_reason == "refusal":
            raise ProviderError("anthropic: request was refused")
        text = "".join(block.text for block in raw.content if block.type == "text")
        return ChatCompletion(
            model=raw.model,
            content=text.strip(),
            usage=Usage(
                input_tokens=raw.usage.input_tokens,
                output_tokens=raw.usage.output_tokens,
            ),
            finish_reason=raw.stop_reason,
        )

    async def aclose(self) -> None:
        await self._client.close()


def _to_anthropic_content(m: Message) -> str | list[dict[str, Any]]:
    """Our parts -> Anthropic content blocks (images go as base64 `image` blocks)."""
    if isinstance(m.content, str):
        return m.content
    blocks: list[dict[str, Any]] = []
    for part in m.content:
        if isinstance(part, ImagePart):
            blocks.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": part.media_type,
                        "data": part.data,
                    },
                }
            )
        else:
            blocks.append({"type": "text", "text": part.text})
    return blocks
