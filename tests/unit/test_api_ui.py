"""API parts used by the web UI: evidence in /query, /documents, /results, static UI."""

import json
from pathlib import Path

import polars as pl
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


def _service(store: QdrantStore, embeddings: EmbeddingService) -> AnswerService:
    config = ExperimentConfig(
        name="t", embedder="fake:64", reranker="fake:x", rerank_threshold=0.3
    )
    retriever = Retriever(embeddings, store, config, FakeReranker())
    return AnswerService(retriever, LLMProxy({"fake": FakeProvider()}), "fake:m")


def _results(tmp_path: Path) -> Path:
    """A minimal results/ folder with one answer run."""
    run = tmp_path / "results" / "20260101T000000_demo_abc123"
    run.mkdir(parents=True)
    pl.DataFrame({"slice": ["all"], "n": [3]}).write_csv(run / "metrics.csv")
    pl.DataFrame(
        {
            "qid": ["a", "b", "c"],
            "doc_id": ["x", "x", "scan"],
            "slices": ["text", "text", "image,no_text_layer"],
            "answerable": ["true", "false", "true"],
            "recall@10": ["1.0", "NaN", "0.0"],
            "correct": ["true", "true", "false"],
            "abstained": ["false", "true", "false"],
            "abstained_before_llm": ["false", "true", "false"],
            "faithful": ["1.0", "NaN", "0.0"],
        }
    ).write_csv(run / "per_question.csv")
    (run / "config.yaml").write_text(
        "config_hash: abc123\nname: demo\nretrieval: hybrid_rerank\n"
        "dataset:\n  split: test\n"
    )
    (run / "run.json").write_text(json.dumps({"llm": {"cost_eur_full": 0.5}}))
    return tmp_path / "results"


def test_query_returns_ranked_evidence_and_usage(
    memory_store: QdrantStore, embeddings: EmbeddingService
) -> None:
    app = create_app(_service(memory_store, embeddings))
    with TestClient(app) as client:
        body = client.post("/query", json={"question": QUESTION, "doc_ids": ["tiny"]})
    data = body.json()
    ev = data["evidence"]
    assert [e["rank"] for e in ev] == list(range(1, len(ev) + 1))
    assert ev[0]["type"] == "TABLE" and ev[0]["cited"], "the cited table ranks first"
    assert data["threshold"] == 0.3 and data["top_score"] == ev[0]["score"]
    assert data["input_tokens"] == 100 and data["output_tokens"] == 20  # FakeProvider


def test_documents_lists_only_indexed_pdfs(
    memory_store: QdrantStore, embeddings: EmbeddingService, tmp_path: Path
) -> None:
    (tmp_path / "tiny.pdf").write_bytes(b"%PDF-1.4")  # indexed in memory_store
    (tmp_path / "other.pdf").write_bytes(b"%PDF-1.4")  # not indexed
    app = create_app(_service(memory_store, embeddings), pdf_dir=tmp_path)
    with TestClient(app) as client:
        assert client.get("/documents").json() == ["tiny"]


def test_results_come_from_the_results_folder(
    memory_store: QdrantStore, embeddings: EmbeddingService, tmp_path: Path
) -> None:
    app = create_app(_service(memory_store, embeddings), results_dir=_results(tmp_path))
    with TestClient(app) as client:
        data = client.get("/results").json()
    assert data["split"] == {"questions": 3, "documents": 2, "scanned_documents": 1}
    run = data["runs"][0]
    assert run["name"] == "demo" and run["split"] == "test"
    assert run["correct"]["mean"] == 2 / 3
    assert run["scanned"]["questions"] == 1 and run["text_layer"]["recall_at_10"] == 1.0
    assert run["gate_answerable"] == 0.0 and run["cost_eur_full"] == 0.5


def test_built_ui_is_served_at_root_without_shadowing_the_api(
    memory_store: QdrantStore, embeddings: EmbeddingService, tmp_path: Path
) -> None:
    ui = tmp_path / "dist"
    ui.mkdir()
    (ui / "index.html").write_text("<!doctype html><title>docrag</title>")
    app = create_app(_service(memory_store, embeddings), ui_dir=ui)
    with TestClient(app) as client:
        assert "<title>docrag</title>" in client.get("/").text
        assert client.get("/health").json() == {"status": "ok"}
