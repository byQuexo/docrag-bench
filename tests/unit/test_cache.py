"""Cache keys and the SQLite key-value store."""

from pathlib import Path

import pytest

from docrag.shared.cache import Cache, bytes_hash, stable_hash


def test_stable_hash_ignores_dict_key_order() -> None:
    assert stable_hash({"a": 1, "b": 2}) == stable_hash({"b": 2, "a": 1})


def test_stable_hash_changes_with_content() -> None:
    assert stable_hash("x", 1) != stable_hash("x", 2)
    assert bytes_hash(b"a") != bytes_hash(b"b")


def test_roundtrip_bytes_and_json(tmp_path: Path) -> None:
    cache = Cache(tmp_path / "sub" / "c.sqlite")  # parent dirs are created
    cache.set_bytes("embed", "k", b"\x00\x01")
    cache.set_json("llm", "k", {"answer": "x", "n": [1, 2]})
    assert cache.get_bytes("embed", "k") == b"\x00\x01"
    assert cache.get_json("llm", "k") == {"answer": "x", "n": [1, 2]}
    cache.close()


def test_namespaces_are_separate() -> None:
    cache = Cache(":memory:")
    cache.set_json("parse", "same-key", 1)
    assert cache.get_json("embed", "same-key") is None
    assert cache.count("parse") == 1
    assert cache.count("embed") == 0


def test_overwrite_and_persistence(tmp_path: Path) -> None:
    path = tmp_path / "c.sqlite"
    first = Cache(path)
    first.set_json("ns", "k", "old")
    first.set_json("ns", "k", "new")  # INSERT OR REPLACE
    first.close()
    # A new connection to the same file sees the data: it really is on disk.
    second = Cache(path)
    assert second.get_json("ns", "k") == "new"
    assert second.count("ns") == 1
    second.close()


@pytest.mark.parametrize("missing", ["nope", ""])
def test_missing_key_returns_none(missing: str) -> None:
    assert Cache(":memory:").get_bytes("ns", missing) is None
