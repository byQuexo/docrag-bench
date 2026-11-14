"""Operational endpoints and behaviour: probes, request deadline, access log."""

import asyncio
import json
import logging
import time
from typing import Any

import pytest
from fastapi.testclient import TestClient
from qdrant_client.http.exceptions import ResponseHandlingException

from docrag.answer.service import AnswerService
from docrag.api.app import create_app
from docrag.embed.base import EmbeddingService
from docrag.llm.fake import FakeProvider
from docrag.llm.proxy import LLMProxy
from docrag.rerank.reranker import FakeReranker
from docrag.shared.config import ExperimentConfig
from docrag.shared.logs import JsonFormatter
from docrag.store.qdrant import QdrantStore, make_client
from docrag.store.retriever import Retriever

QUESTION = "What was the revenue in Europe in 2024?"


def _service(
    store: QdrantStore, embeddings: EmbeddingService, provider: FakeProvider
) -> AnswerService:
    config = ExperimentConfig(
        name="t", embedder="fake:64", reranker="fake:x", rerank_threshold=0.3
    )
    retriever = Retriever(embeddings, store, config, FakeReranker())
    return AnswerService(retriever, LLMProxy({"fake": provider}), "fake:m")


class SlowProvider(FakeProvider):
    """An LLM that takes longer than the request deadline."""

    async def _call(self, native: dict[str, Any]) -> Any:
        await asyncio.sleep(5)
        return await super()._call(native)


def test_readyz_ok_when_collection_exists(
    memory_store: QdrantStore, embeddings: EmbeddingService
) -> None:
    app = create_app(_service(memory_store, embeddings, FakeProvider()))
    with TestClient(app) as client:
        r = client.get("/readyz")
    assert r.status_code == 200
    assert r.json() == {"status": "ready", "collection": "test_chunks"}


def test_readyz_503_when_collection_missing(embeddings: EmbeddingService) -> None:
    store = QdrantStore(make_client(":memory:"), "never_created", embeddings.dim)
    app = create_app(_service(store, embeddings, FakeProvider()))
    with TestClient(app) as client:
        r = client.get("/readyz")
        # Liveness stays green: restarting the API would not create the index.
        assert client.get("/healthz").status_code == 200
    assert r.status_code == 503
    assert "never_created" in r.json()["reason"]


def test_readyz_503_when_qdrant_unreachable(
    memory_store: QdrantStore,
    embeddings: EmbeddingService,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def down() -> bool:
        raise ConnectionError("connection refused")

    monkeypatch.setattr(memory_store, "ping", down)
    app = create_app(_service(memory_store, embeddings, FakeProvider()))
    with TestClient(app) as client:
        r = client.get("/readyz")
    assert r.status_code == 503
    assert "ConnectionError" in r.json()["reason"]


def test_readyz_503_when_qdrant_hangs(
    memory_store: QdrantStore,
    embeddings: EmbeddingService,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def hang() -> bool:
        time.sleep(0.5)
        return True

    monkeypatch.setattr(memory_store, "ping", hang)
    app = create_app(
        _service(memory_store, embeddings, FakeProvider()), readiness_timeout_s=0.05
    )
    with TestClient(app) as client:
        r = client.get("/readyz")
    assert r.status_code == 503
    assert "no answer within" in r.json()["reason"]


def test_query_deadline_returns_504(
    memory_store: QdrantStore, embeddings: EmbeddingService
) -> None:
    app = create_app(
        _service(memory_store, embeddings, SlowProvider()), request_timeout_s=0.2
    )
    with TestClient(app) as client:
        r = client.post("/query", json={"question": QUESTION, "doc_ids": ["tiny"]})
    assert r.status_code == 504
    assert "0.2 s" in r.json()["detail"]


def test_access_log_has_fields_and_skips_probes(
    memory_store: QdrantStore,
    embeddings: EmbeddingService,
    caplog: pytest.LogCaptureFixture,
) -> None:
    app = create_app(_service(memory_store, embeddings, FakeProvider()))
    with caplog.at_level(logging.INFO, logger="docrag.access"), TestClient(app) as c:
        c.post("/query", json={"question": QUESTION, "doc_ids": ["tiny"]})
        c.get("/healthz")
        c.get("/readyz")
    lines = [r for r in caplog.records if r.name == "docrag.access"]
    assert len(lines) == 1  # probes are not logged
    record = json.loads(JsonFormatter().format(lines[0]))
    assert record["method"] == "POST"
    assert record["path"] == "/query"
    assert record["status"] == 200
    assert record["duration_ms"] > 0


def test_qdrant_outage_during_query_is_503(
    memory_store: QdrantStore,
    embeddings: EmbeddingService,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def down(**kwargs: Any) -> Any:
        raise ResponseHandlingException(ConnectionError("refused"))

    monkeypatch.setattr(memory_store, "search", down)
    app = create_app(_service(memory_store, embeddings, FakeProvider()))
    with TestClient(app) as client:
        r = client.post("/query", json={"question": QUESTION, "doc_ids": ["tiny"]})
    assert r.status_code == 503
    assert r.json()["detail"] == "search backend unavailable"
