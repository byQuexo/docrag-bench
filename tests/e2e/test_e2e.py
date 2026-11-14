"""End-to-end: real parser + real Qdrant server + real FastAPI app.

Only the models are faked (embedder, reranker, LLM), so CI needs no downloads
or API keys. Qdrant runs in a throwaway container started by testcontainers
(same image tag as compose.yaml), so the test needs nothing but Docker:

    uv run pytest -m e2e

Why testcontainers instead of a CI service container: the same command works
on a laptop and in CI, each run gets a fresh server, and the container is
removed afterwards even if a test fails. Without Docker the tests are skipped.
"""

import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from qdrant_client.http.exceptions import UnexpectedResponse

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

# Keep in sync with compose.yaml and deploy/k8s: test what is deployed.
QDRANT_IMAGE = "qdrant/qdrant:v1.20.0"

pytestmark = pytest.mark.e2e


def _docker_available() -> bool:
    try:
        import docker

        docker.from_env().ping()
        return True
    except Exception:
        return False


@pytest.fixture(scope="module")
def qdrant_url() -> Iterator[str]:
    if not _docker_available():
        pytest.skip("Docker is not running")
    from testcontainers.community.qdrant import QdrantContainer

    # `with` stops and removes the container, also when a test failed.
    with QdrantContainer(QDRANT_IMAGE) as qdrant:
        yield f"http://{qdrant.rest_host_address}"


@pytest.fixture(scope="module")
def store(qdrant_url: str) -> Iterator[QdrantStore]:
    client = make_client(qdrant_url)
    name = f"e2e_{uuid.uuid4().hex[:8]}"
    store = QdrantStore(client, name, dim=64)
    store.ensure_collection()
    yield store
    client.delete_collection(name)


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


def test_readyz_against_a_real_qdrant(client: TestClient) -> None:
    r = client.get("/readyz")
    assert r.status_code == 200, r.text
    assert r.json()["status"] == "ready"


def test_missing_collection_fails_fast_without_retries(qdrant_url: str) -> None:
    """A 404 is a config error, not a hiccup: no backoff, the error at once."""
    import time

    store = QdrantStore(make_client(qdrant_url), "does_not_exist", dim=64, retries=3)
    start = time.perf_counter()
    with pytest.raises(UnexpectedResponse) as err:
        store.count()
    assert err.value.status_code == 404
    assert time.perf_counter() - start < 1.0
    assert store.ping() is False
