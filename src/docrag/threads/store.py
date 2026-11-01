"""Conversation threads: interface + in-memory implementation (Redis later).

The API is stateless; all conversation state lives behind this interface. Any
API process can serve any request once the store is shared (Redis), which is
what makes horizontal scaling possible.
"""

import threading
import uuid
from typing import Protocol

from docrag.llm.types import Message


class ThreadStore(Protocol):
    def create(self) -> str: ...
    def get(self, thread_id: str) -> list[Message] | None: ...
    def append(self, thread_id: str, *messages: Message) -> None: ...


class InMemoryThreadStore:
    """Dict-backed store. Lost on restart; fine for one process and for tests."""

    def __init__(self, max_messages: int = 20) -> None:
        self._threads: dict[str, list[Message]] = {}
        self._lock = threading.Lock()  # FastAPI may call us from several threads
        self.max_messages = max_messages

    def create(self) -> str:
        thread_id = uuid.uuid4().hex
        with self._lock:
            self._threads[thread_id] = []
        return thread_id

    def get(self, thread_id: str) -> list[Message] | None:
        with self._lock:
            msgs = self._threads.get(thread_id)
            # Return a copy: callers must not mutate our internal list.
            return None if msgs is None else list(msgs)

    def append(self, thread_id: str, *messages: Message) -> None:
        with self._lock:
            if thread_id not in self._threads:
                raise KeyError(thread_id)
            msgs = self._threads[thread_id]
            msgs.extend(messages)
            # Keep only the last N messages so prompts don't grow without bound.
            del msgs[: max(0, len(msgs) - self.max_messages)]
