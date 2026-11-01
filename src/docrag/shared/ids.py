"""Stable chunk IDs like ``doc3:p2:c1``.

The ID is the backbone of the citation flow: the LLM sees ``[doc3:p2:c1]`` in its
context, cites it, and Verify checks that exact string. So the ID must be:

* stable across runs (same PDF + same parser/chunker version -> same IDs),
* human readable (you can see the page in it),
* safe to embed in prompts (no ``:``, ``[`` or ``]`` inside the doc part).
"""

import re
import uuid

_VALID_DOC_ID = re.compile(r"[A-Za-z0-9._-]+")
_CHUNK_ID = re.compile(r"(?P<doc>[A-Za-z0-9._-]+):p(?P<page>\d+):c(?P<idx>\d+)")

# Qdrant point IDs must be an int or a UUID. uuid5 is deterministic, so the
# same chunk ID always maps to the same point: upserts are idempotent.
_POINT_NAMESPACE = uuid.UUID("6f1c3b0e-6a52-4f1a-9a57-2f0f2d7f9b11")


def make_chunk_id(doc_id: str, page: int, index: int) -> str:
    """Build a stable chunk ID like 'doc3:p2:c1'."""
    if not is_valid_doc_id(doc_id):
        raise ValueError(f"Invalid doc_id: {doc_id!r}")
    if page <= 0 or index < 0:
        raise ValueError("page must be >= 1 and index must be >= 0")

    return f"{doc_id}:p{page}:c{index}"


def is_valid_doc_id(doc_id: str) -> bool:
    # fullmatch: the whole string must match (re.match only anchors at the start).
    return _VALID_DOC_ID.fullmatch(doc_id) is not None


def parse_chunk_id(chunk_id: str) -> tuple[str, int, int]:
    """Inverse of make_chunk_id: 'doc3:p2:c1' -> ('doc3', 2, 1)."""
    m = _CHUNK_ID.fullmatch(chunk_id)
    if m is None:
        raise ValueError(f"Invalid chunk_id: {chunk_id!r}")
    return m["doc"], int(m["page"]), int(m["idx"])


def chunk_point_id(chunk_id: str) -> str:
    """Deterministic Qdrant point ID (UUID string) for a chunk ID."""
    return str(uuid.uuid5(_POINT_NAMESPACE, chunk_id))
