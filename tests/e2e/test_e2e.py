"""End-to-end: real parser + real Qdrant server + real FastAPI app.

Only the models are faked (embedder, reranker, LLM), so CI needs no downloads
or API keys. Runs only when DOCRAG_E2E_QDRANT_URL is set, e.g. in CI against
the Qdrant service container, or locally:

    docker compose up -d qdrant
    $env:DOCRAG_E2E_QDRANT_URL = "http://localhost:6333"   # PowerShell
    uv run pytest -m e2e
"""

import os
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from docrag.answer.service import AnswerService
from docrag.api.app import create_app
from docrag.embed.base import EmbeddingService
from docrag.embed.fake import FakeEmbedder
from docrag.ingest.indexer import index_documents
from docrag.llm.fake import FakeProvider
from docrag.llm.proxy import LLMProxy
from docrag.rerank.reranker import FakeReranker
from docrag.shared.config import ExperimentConfig
from docrag.store.qdrant import QdrantStore, make_client
from docrag.store.retriever import Retriever

QDRANT_URL = os.environ.get("DOCRAG_E2E_QDRANT_URL")

# LEARN: `pytestmark` applies marks to every test in the module. skipif turns
# the whole file into "skipped" (not "failed") when there's no Qdrant around.
pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(not QDRANT_URL, reason="set DOCRAG_E2E_QDRANT_URL"),
]


@pytest.fixture(scope="module")
def store() -> Iterator[QdrantStore]:
    assert QDRANT_URL is not None
    client = make_client(QDRANT_URL)
    # Unique name: parallel CI jobs or a crashed earlier run can't collide.
    name = f"e2e_{uuid.uuid4().hex[:8]}"
    store = QdrantStore(client, name, dim=64)
    store.ensure_collection()
    yield store
    client.delete_collection(name)  # teardown runs even if tests failed


@pytest.fixture(scope="module")
def client(store: QdrantStore, tiny_pdf: Path) -> Iterator[TestClient]:
    embeddings = EmbeddingService(FakeEmbedder(dim=64))
    added = index_documents(
        {"tiny": tiny_pdf},
        store,
        embeddings,
        prefix_headings=True,
        cache=None,
        workers=1,
    )
    assert added == 4
    # Indexing again is a no-op (the doc is already in the collection).
    assert (
        index_documents(
            {"tiny": tiny_pdf}, store, embeddings, prefix_headings=True, cache=None
        )
        == 0
    )

    config = ExperimentConfig(
        name="e2e", embedder="fake:64", reranker="fake:x", rerank_threshold=0.3
    )
    retriever = Retriever(embeddings, store, config, FakeReranker())
    service = AnswerService(retriever, LLMProxy({"fake": FakeProvider()}), "fake:m")
    with TestClient(create_app(service, pdf_dir=tiny_pdf.parent)) as c:
        yield c


def test_query_returns_verified_source_on_gold_page(
    client: TestClient, tiny_questions: list[dict[str, Any]]
) -> None:
    q = next(q for q in tiny_questions if "Europe" in q["question"])
    r = client.post("/query", json={"question": q["question"], "doc_ids": ["tiny"]})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "supported"
    assert body["rejected_citations"] == 0
    assert q["evidence_pages"][0] in [s["page"] for s in body["sources"]]

    image = client.get(body["sources"][0]["page_image_url"])
    assert image.status_code == 200 and image.content.startswith(b"\x89PNG")


@pytest.mark.parametrize("idx", [0, 1, 2])
def test_every_fixture_question_retrieves_its_page(
    store: QdrantStore, tiny_questions: list[dict[str, Any]], idx: int
) -> None:
    q = tiny_questions[idx]
    embeddings = EmbeddingService(FakeEmbedder(dim=64))
    hits = store.search(
        query_vector=embeddings.embed_query(q["question"]),
        query_text=q["question"],
        mode="hybrid",
        limit=3,
        doc_ids=["tiny"],
    )
    pages = {p for h in hits for p in h.chunk.pages}
    assert set(q["evidence_pages"]) <= pages
