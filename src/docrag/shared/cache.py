"""A tiny persistent key-value cache on SQLite, with versioned keys.

Why a cache at all? Parsing 130 PDFs, embedding ~50k chunks and paying for LLM
answers are the three slow/expensive steps. With the cache, a second benchmark run
costs 0 EUR and takes minutes, which is a milestone criterion (M5).

The important part is the KEY, not the storage:

    key = namespace + component version + hash(content) + result-relevant options

* content hash, not file path or mtime: renaming a PDF must not invalidate, editing
  it must.
* component version: when the parser logic changes, bump ``PARSER_VERSION`` and
  every old entry is simply never looked up again (no manual "clear cache").
* options: anything that changes the output (e.g. heading prefix on/off).
* never secrets (API keys) and never streams.
"""

import hashlib
import json
import sqlite3
import threading
from pathlib import Path
from typing import Any


def stable_hash(*parts: Any) -> str:
    """SHA-256 over a canonical JSON encoding of ``parts``.

    ``sort_keys=True`` matters: {"a":1,"b":2} and {"b":2,"a":1} must hash the same.
    """
    payload = json.dumps(parts, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def bytes_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class Cache:
    """Namespaced key-value store. Values are bytes or JSON-serializable objects."""

    def __init__(self, path: Path | str) -> None:
        self._path = str(path)
        if self._path != ":memory:":
            Path(self._path).parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False: FastAPI and asyncio.to_thread may touch the cache
        # from worker threads. The lock below makes that safe.
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock:
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS kv ("
                " ns TEXT NOT NULL, key TEXT NOT NULL, value BLOB NOT NULL,"
                " PRIMARY KEY (ns, key))"
            )
            # WAL = better concurrent reads, fewer "database is locked" errors.
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.commit()

    def get_bytes(self, ns: str, key: str) -> bytes | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT value FROM kv WHERE ns = ? AND key = ?", (ns, key)
            ).fetchone()
        # Parameter binding only: never build SQL from data with f-strings.
        return None if row is None else bytes(row[0])

    def set_bytes(self, ns: str, key: str, value: bytes) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO kv (ns, key, value) VALUES (?, ?, ?)",
                (ns, key, value),
            )
            self._conn.commit()

    def get_json(self, ns: str, key: str) -> Any | None:
        raw = self.get_bytes(ns, key)
        return None if raw is None else json.loads(raw)

    def set_json(self, ns: str, key: str, value: Any) -> None:
        self.set_bytes(ns, key, json.dumps(value, ensure_ascii=False).encode("utf-8"))

    def count(self, ns: str) -> int:
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) FROM kv WHERE ns = ?", (ns,)
            ).fetchone()
        return int(row[0])

    def close(self) -> None:
        with self._lock:
            self._conn.close()
