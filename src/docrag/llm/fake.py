"""FakeProvider: a deterministic "LLM" for unit tests, the E2E test and CI.

It reads the context lines ``[id] text`` from the prompt and answers by citing
the first chunk with a real quote, so the full pipeline (including Verify) can
be exercised without a model or an API key.
"""

import json
import re
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel

from docrag.llm.base import Provider
from docrag.llm.types import ChatCompletion, Message, Usage

# "[doc:p1:c0] (page 1, text, section: Intro) The actual chunk text ..."
_CONTEXT_LINE = re.compile(
    r"^\[(?P<id>[^\]\s]+)\]\s*\([^)]*\)\s*(?P<text>.+)$", re.MULTILINE
)

Responder = Callable[[list[Message], type[BaseModel] | None], str]


def default_responder(messages: list[Message], schema: type[BaseModel] | None) -> str:
    prompt = "\n".join(m.text for m in messages)
    name = schema.__name__ if schema else ""
    if name == "Answer":
        match = _CONTEXT_LINE.search(prompt)
        if match is None:
            return json.dumps({"answer": "Not answerable", "evidence": []})
        quote = " ".join(match["text"].split()[:8])
        cid = match["id"]
        return json.dumps(
            {"answer": f"{quote} [{cid}]", "evidence": [{"id": cid, "quote": quote}]}
        )
    if name == "JudgeVerdict":
        return json.dumps(
            {"reasoning": "fake judge", "correct": True, "faithful": True}
        )
    return "ok"


class FakeProvider(Provider):
    name = "fake"

    def __init__(self, responder: Responder | None = None) -> None:
        self.responder = responder or default_responder
        self.calls: list[dict[str, Any]] = []  # tests can inspect what was sent

    def to_native(
        self,
        model: str,
        messages: list[Message],
        response_format: type[BaseModel] | None,
        temperature: float | None,
    ) -> dict[str, Any]:
        return {"model": model, "messages": messages, "schema": response_format}

    async def _call(self, native: dict[str, Any]) -> Any:
        self.calls.append(native)
        return self.responder(native["messages"], native["schema"])

    def from_native(self, model: str, raw: Any) -> ChatCompletion:
        return ChatCompletion(
            model=model, content=raw, usage=Usage(input_tokens=100, output_tokens=20)
        )
