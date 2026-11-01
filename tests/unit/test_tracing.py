"""Tracing: spans only inside a trace, one connected trace per question.

All tests run against an InMemorySpanExporter (fixture `spans` in conftest.py),
so no collector is needed.
"""

from pathlib import Path

from fastapi.testclient import TestClient
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

from docrag.answer.service import AnswerService
from docrag.api.app import create_app
from docrag.embed.base import EmbeddingService
from docrag.ingest.pipeline import ingest_pdf
from docrag.llm.fake import FakeProvider
from docrag.llm.proxy import LLMProxy
from docrag.rerank.reranker import FakeReranker
from docrag.shared import timing, tracing
from docrag.shared.config import ExperimentConfig
from docrag.store.qdrant import QdrantStore
from docrag.store.retriever import Retriever

QUESTION = "What was the revenue in Europe in 2024?"


def _service(store: QdrantStore, embeddings: EmbeddingService) -> AnswerService:
    config = ExperimentConfig(
        name="t", embedder="fake:64", reranker="fake:x", rerank_threshold=0.3
    )
    retriever = Retriever(embeddings, store, config, FakeReranker())
    return AnswerService(retriever, LLMProxy({"fake": FakeProvider()}), "fake:m")


def _by_name(spans: InMemorySpanExporter) -> dict[str, ReadableSpan]:
    return {s.name: s for s in spans.get_finished_spans()}


@timing.timed("unit_step")
def _step() -> int:
    return 42


def test_timed_outside_a_trace_records_duration_but_no_span(
    spans: InMemorySpanExporter,
) -> None:
    timing.reset()
    assert _step() == 42
    assert spans.get_finished_spans() == ()
    assert timing.percentiles("unit_step")["n"] == 1.0  # the benchmark still counts


def test_timed_inside_a_trace_becomes_a_child_span(spans: InMemorySpanExporter) -> None:
    with tracing.root_span("root"):
        _step()
    named = _by_name(spans)
    child, root = named["unit_step"], named["root"]
    assert child.parent is not None
    assert child.parent.span_id == root.context.span_id
    assert child.context.trace_id == root.context.trace_id


def test_ingest_produces_no_spans(
    spans: InMemorySpanExporter, tiny_pdf: Path, embeddings: EmbeddingService
) -> None:
    # parse + embed are @timed, but ingest has no root span -> nothing recorded.
    chunks = ingest_pdf(tiny_pdf, "tiny")
    embeddings.embed([c.embed_text for c in chunks])
    assert spans.get_finished_spans() == ()


async def test_answer_is_one_connected_trace(
    spans: InMemorySpanExporter,
    memory_store: QdrantStore,
    embeddings: EmbeddingService,
) -> None:
    # No root around it: AnswerService.answer is root=True (`docrag ask` case).
    result, _ = await _service(memory_store, embeddings).answer(
        QUESTION, doc_ids=["tiny"]
    )
    named = _by_name(spans)
    expected = {"answer", "retrieve", "embed", "search", "rerank", "llm", "verify"}
    assert expected <= set(named)
    # One trace, and only the answer span is a root.
    assert len({s.context.trace_id for s in named.values()}) == 1
    assert named["answer"].parent is None
    # retrieve ran in a worker thread (asyncio.to_thread) and still has its parent.
    retrieve_parent = named["retrieve"].parent
    assert retrieve_parent is not None
    assert retrieve_parent.span_id == named["answer"].context.span_id

    answer_attrs = named["answer"].attributes or {}
    assert answer_attrs[tracing.SPAN_KIND] == tracing.CHAIN
    assert answer_attrs["docrag.status"] == result.status

    llm = named["llm"].attributes or {}
    assert llm[tracing.SPAN_KIND] == tracing.LLM
    assert llm["llm.model_name"] == "m"
    assert llm["llm.input_messages.0.message.role"] == "system"
    assert "llm.token_count.prompt" in llm

    retrieved = named["retrieve"].attributes or {}
    assert str(retrieved["retrieval.documents.0.document.id"]).startswith("tiny:")
    assert (named["verify"].attributes or {})["docrag.verify.status"] == result.status


def test_child_of_joins_an_already_ended_trace(spans: InMemorySpanExporter) -> None:
    with tracing.root_span("question") as span:
        parent = span.get_span_context()
    # Later, after the question span ended (like the judge phase in eval):
    with tracing.child_of(parent, "judge", kind=tracing.EVALUATOR):
        pass
    named = _by_name(spans)
    assert named["judge"].context.trace_id == named["question"].context.trace_id


def test_api_request_is_the_trace_root(
    spans: InMemorySpanExporter,
    memory_store: QdrantStore,
    embeddings: EmbeddingService,
) -> None:
    app = create_app(_service(memory_store, embeddings))
    with TestClient(app) as client:
        body = client.post(
            "/query", json={"question": QUESTION, "doc_ids": ["tiny"]}
        ).json()
        client.get("/health")
    named = _by_name(spans)
    # FastAPI's own HTTP server span is the root of the trace ...
    http = named["POST /query"]
    assert http.parent is None
    assert (http.attributes or {})["http.response.status_code"] == 200
    # ... our "query" span is below it, and the answer pipeline below that.
    query = named["query"]
    assert (query.attributes or {})["session.id"] == body["thread_id"]
    assert len({s.context.trace_id for s in named.values()}) == 1
    answer_parent = named["answer"].parent
    assert answer_parent is not None
    assert answer_parent.span_id == query.context.span_id
    assert not any("/health" in s.name for s in spans.get_finished_spans())


def test_setup_without_endpoint_is_off() -> None:
    assert tracing.setup_tracing(None) is False
