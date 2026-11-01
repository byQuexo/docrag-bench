"""FastAPI routes via TestClient, with the fake pipeline behind them."""

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from docrag.answer.service import AnswerService
from docrag.api.app import create_app
from docrag.embed.base import EmbeddingService
from docrag.llm.fake import FakeProvider
from docrag.llm.proxy import LLMProxy
from docrag.rerank.reranker import FakeReranker
from docrag.shared.config import ExperimentConfig
from docrag.store.qdrant import QdrantStore
from docrag.store.retriever import Retriever

QUESTION = "What was the revenue in Europe in 2024?"


@pytest.fixture
def client(
    memory_store: QdrantStore, embeddings: EmbeddingService, tiny_pdf: Path
) -> Iterator[TestClient]:
    config = ExperimentConfig(
        name="t", embedder="fake:64", reranker="fake:x", rerank_threshold=0.3
    )
    retriever = Retriever(embeddings, memory_store, config, FakeReranker())
    service = AnswerService(retriever, LLMProxy({"fake": FakeProvider()}), "fake:m")
    app = create_app(service, pdf_dir=tiny_pdf.parent)
    with TestClient(app) as c:
        yield c


def test_health(client: TestClient) -> None:
    assert client.get("/health").json() == {"status": "ok"}


def test_query_creates_thread_and_history_grows(client: TestClient) -> None:
    r = client.post("/query", json={"question": QUESTION, "doc_ids": ["tiny"]})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "supported"
    assert body["sources"][0]["page_image_url"].startswith("/pages/tiny/")
    thread_id = body["thread_id"]

    r2 = client.post("/query", json={"question": "And Asia?", "thread_id": thread_id})
    assert r2.status_code == 200 and r2.json()["thread_id"] == thread_id
    history = client.get(f"/threads/{thread_id}").json()
    assert [m["role"] for m in history] == ["user", "assistant", "user", "assistant"]
    assert history[0]["content"] == QUESTION


def test_unknown_thread_is_404(client: TestClient) -> None:
    r = client.post("/query", json={"question": "x", "thread_id": "nope"})
    assert r.status_code == 404
    assert client.get("/threads/nope").status_code == 404


def test_request_validation(client: TestClient) -> None:
    # FastAPI validates the body against QueryRequest: empty question -> 422.
    assert client.post("/query", json={"question": ""}).status_code == 422


def test_page_png_is_served(client: TestClient) -> None:
    r = client.get("/pages/tiny/2.png")
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/png"
    assert r.content.startswith(b"\x89PNG")
    assert client.get("/pages/tiny/99.png").status_code == 404


def test_missing_pdf_is_404(client: TestClient) -> None:
    assert client.get("/pages/missing/1.png").status_code == 404


@pytest.mark.parametrize("doc_id", ["..", "%2E%2E%2Fsecrets", "a b"])
def test_page_path_traversal_rejected(client: TestClient, doc_id: str) -> None:
    r = client.get(f"/pages/{doc_id}/1.png")
    assert r.status_code in (400, 404)
    assert not r.content.startswith(b"\x89PNG")


def test_safe_pdf_path_rejects_traversal(tiny_pdf: Path) -> None:
    from docrag.ingest.pages import UnsafePathError, safe_pdf_path

    assert safe_pdf_path(tiny_pdf.parent, "tiny") == tiny_pdf.resolve()
    # ".." alone is NOT traversal here: it becomes the file name "...pdf" inside
    # the folder. Slashes are what would let a path escape.
    for bad in ("../tiny", "a/b", "a\\b", ""):
        with pytest.raises(UnsafePathError):
            safe_pdf_path(tiny_pdf.parent, bad)
