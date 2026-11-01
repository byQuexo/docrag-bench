"""The internal message format: a small subset of the OpenAI chat format.

Everything inside docrag speaks this format. Each provider adapter converts it
to its native API and back (``to_native`` / ``from_native``), so adding a
provider never touches the answer, judge or eval code.
"""

import base64
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field


class TextPart(BaseModel):
    type: Literal["text"] = "text"
    text: str


class ImagePart(BaseModel):
    """An image inside a message (vision models). Stored as base64."""

    type: Literal["image"] = "image"
    media_type: Literal["image/png", "image/jpeg"] = "image/png"
    data: str  # base64, no "data:" prefix

    @classmethod
    def from_png(cls, png: bytes) -> "ImagePart":
        return cls(data=base64.b64encode(png).decode("ascii"))


ContentPart = Annotated[TextPart | ImagePart, Field(discriminator="type")]


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    # Plain text (the common case), or a list of parts for text + images.
    content: str | list[ContentPart]

    @property
    def text(self) -> str:
        """The text of the message, ignoring images."""
        if isinstance(self.content, str):
            return self.content
        return "\n".join(p.text for p in self.content if isinstance(p, TextPart))


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    # Set when the provider reports the billed cost itself (OpenRouter does);
    # otherwise cost is computed from our price table.
    cost_usd: float | None = None


class ChatCompletion(BaseModel):
    model: str
    content: str  # final text only - reasoning/thinking is never put here
    usage: Usage = Usage()
    finish_reason: str | None = None


class ProviderError(RuntimeError):
    """A provider call failed in a way retrying won't fix (refusal, 4xx, ...)."""


def json_schema_for(model: type[BaseModel]) -> dict[str, Any]:
    """JSON schema of a Pydantic model, as sent in response_format.

    Our response models set ``extra="forbid"``, so Pydantic already emits
    ``additionalProperties: false``, which strict structured output requires.
    """
    return model.model_json_schema()
