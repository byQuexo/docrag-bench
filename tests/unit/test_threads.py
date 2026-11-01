"""InMemoryThreadStore."""

import pytest

from docrag.llm.types import Message
from docrag.threads.store import InMemoryThreadStore


def _m(i: int) -> Message:
    return Message(role="user", content=str(i))


def test_create_append_get() -> None:
    store = InMemoryThreadStore()
    tid = store.create()
    assert store.get(tid) == []
    store.append(tid, _m(1), _m(2))
    assert [m.content for m in store.get(tid) or []] == ["1", "2"]
    assert store.get("unknown") is None


def test_get_returns_a_copy() -> None:
    store = InMemoryThreadStore()
    tid = store.create()
    store.get(tid).append(_m(1))  # type: ignore[union-attr]
    assert store.get(tid) == []


def test_append_to_unknown_thread() -> None:
    with pytest.raises(KeyError):
        InMemoryThreadStore().append("nope", _m(1))


def test_history_is_capped() -> None:
    store = InMemoryThreadStore(max_messages=3)
    tid = store.create()
    store.append(tid, *[_m(i) for i in range(5)])
    assert [m.content for m in store.get(tid) or []] == ["2", "3", "4"]


def test_thread_ids_are_unique() -> None:
    store = InMemoryThreadStore()
    assert len({store.create() for _ in range(100)}) == 100
