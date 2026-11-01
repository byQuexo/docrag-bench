"""The Gradio demo's glue code, offline with fakes (skipped without the demo extra)."""

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

pytest.importorskip("gradio")

from docrag.answer.schema import AnswerResult, Citation  # noqa: E402
from docrag.answer.service import AnswerService  # noqa: E402
from docrag.embed.base import EmbeddingService  # noqa: E402
from docrag.embed.fake import FakeEmbedder  # noqa: E402
from docrag.llm.fake import FakeProvider  # noqa: E402
from docrag.llm.proxy import LLMProxy  # noqa: E402
from docrag.rerank.reranker import FakeReranker  # noqa: E402
from docrag.shared.config import ExperimentConfig  # noqa: E402
from docrag.shared.settings import Settings  # noqa: E402
from docrag.store.qdrant import QdrantStore, make_client  # noqa: E402
from docrag.store.retriever import Retriever  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def app() -> ModuleType:
    spec = importlib.util.spec_from_file_location("demo_app", ROOT / "demo" / "app.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _result(status: str, answer: str, **kw: object) -> AnswerResult:
    return AnswerResult(
        question="q",
        answer=answer,
        raw_answer=answer,
        status=status,  # type: ignore[arg-type]
        citations=kw.pop("citations", []),  # type: ignore[arg-type]
        context_ids=[],
        **kw,  # type: ignore[arg-type]
    )


def test_gate_abstention_explains_the_threshold(app: ModuleType) -> None:
    md, rows = app.render_answer(
        _result("abstained", "Not answerable", abstained_before_llm=True), [], 0.58
    )
    assert "Not answerable from this document" in md and "0.58" in md
    assert rows == []


def test_llm_abstention_and_verified_answer(app: ModuleType) -> None:
    md, _ = app.render_answer(_result("abstained", "Not answerable"), [], 0.58)
    assert "Not answerable" in md and "threshold" not in md
    cit = Citation(id="d:p12:c1", quote="4,821", doc_id="d", page=12, verified=True)
    md, _ = app.render_answer(
        _result("supported", "Revenue was 4,821 [d:p12:c1].", citations=[cit]), [], 0.58
    )
    assert "**[p.12]**" in md and "[d:p12:c1]" not in md
    assert "every citation verified" in md and "4,821" in md


async def test_ask_runs_the_real_pipeline_with_fakes(app: ModuleType) -> None:
    config = ExperimentConfig(
        name="t", embedder="fake:64", reranker="fake:x", rerank_threshold=0.3
    )
    embeddings = EmbeddingService(FakeEmbedder(dim=64))
    store = QdrantStore(make_client(":memory:"), "demo_test", embeddings.dim)
    store.ensure_collection()
    service = AnswerService(
        Retriever(embeddings, store, config, FakeReranker()),
        LLMProxy({"fake": FakeProvider()}),
        "fake:m",
    )
    app._state.update(
        settings=Settings(qdrant_location=":memory:", use_keyring=False),
        service=service,
        retriever=service.retriever,
        config=config,
        ocr=None,
        cache=None,
    )
    md, rows, image, asked = await app.ask(
        "tiny", None, "What was the revenue in Europe in 2024?", 0
    )
    assert asked == 1 and rows, "indexed on demand, evidence returned"
    assert "[p." in md and "Verify:" in md
    assert image is not None, "page image of the first verified citation"
    with pytest.raises(app.gr.Error):
        await app.ask("tiny", None, "again", app.MAX_QUESTIONS)  # session limit


async def test_retrieval_only_mode_shows_evidence_without_an_answer(
    app: ModuleType,
) -> None:
    config = ExperimentConfig(
        name="t", embedder="fake:64", reranker="fake:x", rerank_threshold=0.3
    )
    embeddings = EmbeddingService(FakeEmbedder(dim=64))
    store = QdrantStore(make_client(":memory:"), "demo_ro", embeddings.dim)
    store.ensure_collection()
    app._state.update(
        settings=Settings(qdrant_location=":memory:", use_keyring=False),
        service=None,  # no answer model configured
        retriever=Retriever(embeddings, store, config, FakeReranker()),
        config=config,
        ocr=None,
        cache=None,
    )
    md, rows, image, _ = await app.ask("tiny", None, "Revenue in Europe 2024?", 0)
    assert "Answer generation is off" in md
    assert rows and rows[0][2] == "TABLE" and all(r[4] == "" for r in rows)
    assert image is not None, "page of the best hit"
