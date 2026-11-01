"""Shared pytest fixtures. Expensive ones (parsing the PDF) are session-scoped."""

import json
from pathlib import Path
from typing import Any

import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

from docrag.embed.base import EmbeddingService
from docrag.embed.fake import FakeEmbedder
from docrag.ingest.pipeline import ingest_pdf
from docrag.llm.fake import FakeProvider
from docrag.llm.proxy import LLMProxy
from docrag.shared.schema import Chunk
from docrag.shared.tracing import setup_tracing
from docrag.store.qdrant import QdrantStore, make_client

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "fixtures"


@pytest.fixture(scope="session")
def tiny_pdf() -> Path:
    return FIXTURES / "tiny.pdf"


@pytest.fixture(scope="session")
def tiny_questions() -> list[dict[str, Any]]:
    data: list[dict[str, Any]] = json.loads(
        (FIXTURES / "tiny_questions.json").read_text(encoding="utf-8")
    )
    return data


@pytest.fixture(scope="session")
def tiny_chunks(tiny_pdf: Path) -> list[Chunk]:
    return ingest_pdf(tiny_pdf, "tiny")


@pytest.fixture
def embeddings() -> EmbeddingService:
    return EmbeddingService(FakeEmbedder(dim=64), cache=None, batch_size=8)


@pytest.fixture
def memory_store(tiny_chunks: list[Chunk], embeddings: EmbeddingService) -> QdrantStore:
    """An in-process Qdrant (":memory:") already holding tiny.pdf's chunks."""
    store = QdrantStore(make_client(":memory:"), "test_chunks", embeddings.dim)
    store.ensure_collection()
    store.upsert(tiny_chunks, embeddings.embed([c.embed_text for c in tiny_chunks]))
    return store


@pytest.fixture
def fake_provider() -> FakeProvider:
    return FakeProvider()


@pytest.fixture
def fake_proxy(fake_provider: FakeProvider) -> LLMProxy:
    return LLMProxy({"fake": fake_provider}, cache=None, budget_eur=1.0)


@pytest.fixture(scope="session")
def _tracing_exporter() -> InMemorySpanExporter:
    # The global TracerProvider can only be set ONCE per process, so the
    # exporter is session-wide; `spans` below clears it for every test.
    exporter = InMemorySpanExporter()
    setup_tracing(None, exporter=exporter)
    return exporter


@pytest.fixture
def spans(_tracing_exporter: InMemorySpanExporter) -> InMemorySpanExporter:
    """Tracing ON for this test; read finished spans with spans.get_finished_spans()."""
    _tracing_exporter.clear()
    return _tracing_exporter
