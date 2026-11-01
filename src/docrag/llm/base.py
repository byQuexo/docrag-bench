"""Provider base class: a provider only CONVERTS formats and makes the HTTP call.

Routing, caching, cost, retries on invalid JSON and (later) tracing live in the
proxy, written once. A new provider = to_native + _call + from_native.
"""

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel

from docrag.llm.types import ChatCompletion, Message


class Provider(ABC):
    # LEARN: ABC + @abstractmethod = an abstract class (Java `abstract class`).
    # Instantiating a subclass that forgot to implement a method raises TypeError
    # immediately. We use nominal inheritance here (unlike the Parser Protocol)
    # because Provider has a real shared method, chat(), the "template method".

    name: str

    async def chat(
        self,
        model: str,
        messages: list[Message],
        response_format: type[BaseModel] | None = None,
        temperature: float | None = 0.0,
    ) -> ChatCompletion:
        native = self.to_native(model, messages, response_format, temperature)
        raw = await self._call(native)
        return self.from_native(model, raw)

    @abstractmethod
    def to_native(
        self,
        model: str,
        messages: list[Message],
        response_format: type[BaseModel] | None,
        temperature: float | None,
    ) -> dict[str, Any]: ...

    @abstractmethod
    async def _call(self, native: dict[str, Any]) -> Any: ...

    @abstractmethod
    def from_native(self, model: str, raw: Any) -> ChatCompletion: ...

    async def aclose(self) -> None:  # noqa: B027 - optional hook, empty on purpose
        """Release HTTP connections. Override if the provider holds a client."""
