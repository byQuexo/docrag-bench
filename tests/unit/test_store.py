"""QdrantStore against Qdrant's in-process ":memory:" mode (no Docker needed)."""

from typing import Any

import pytest

from docrag.embed.base import EmbeddingService
from docrag.shared.schema import Chunk
from docrag.store.qdrant import QdrantStore, collection_name

QUESTION = "What was the revenue in Europe in 2024?"


@pytest.mark.parametrize("mode", ["dense", "bm25", "hybrid"])
def test_revenue_question_finds_table(
    mode: Any, memory_store: QdrantStore, embeddings: EmbeddingService
) -> None:
    hits = memory_store.search(
        query_vector=embeddings.embed_query(QUESTION),
        query_text=QUESTION,
        mode=mode,
        limit=2,
    )
    assert "tiny:p2:c1" in [h.chunk.id for h in hits], mode
    assert hits[0].chunk.pages == [2]


def test_doc_filter(memory_store: QdrantStore, embeddings: EmbeddingService) -> None:
    vec = embeddings.embed_query(QUESTION)

    def search(doc_ids: list[str]) -> list[str]:
        hits = memory_store.search(
            query_vector=vec, query_text=QUESTION, mode="hybrid", doc_ids=doc_ids
        )
        return [h.chunk.doc_id for h in hits]

    assert set(search(["tiny"])) == {"tiny"}
    assert search(["other"]) == []


def test_upsert_is_idempotent(
    memory_store: QdrantStore,
    tiny_chunks: list[Chunk],
    embeddings: EmbeddingService,
) -> None:
    before = memory_store.count()
    memory_store.upsert(tiny_chunks, embeddings.embed([c.text for c in tiny_chunks]))
    assert memory_store.count() == before == len(tiny_chunks)
    assert memory_store.count("tiny") == len(tiny_chunks)


def test_upsert_length_mismatch(
    memory_store: QdrantStore, tiny_chunks: list[Chunk], embeddings: EmbeddingService
) -> None:
    with pytest.raises(ValueError):
        memory_store.upsert(tiny_chunks, embeddings.embed(["only one"]))


def test_hit_payload_roundtrips_to_chunk(
    memory_store: QdrantStore, tiny_chunks: list[Chunk]
) -> None:
    hits = memory_store.search(
        query_vector=None, query_text="recommends support team Asia", mode="bm25"
    )
    by_id = {c.id: c for c in tiny_chunks}
    assert hits[0].chunk == by_id[hits[0].chunk.id]
    assert hits[0].chunk.id == "tiny:p3:c0"
    assert hits[0].source == "bm25"


def test_bm25_without_terms_returns_nothing(memory_store: QdrantStore) -> None:
    hits = memory_store.search(query_vector=None, query_text="the of", mode="bm25")
    assert hits == []


def test_collection_name_is_safe() -> None:
    name = collection_name("onnx:gte-small", 384, True)
    assert name == "chunks__onnx-gte-small__384__pfx"
    assert collection_name("x/y", 8, False).endswith("__raw")
