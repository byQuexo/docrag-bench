"""Regression tests for the fixes from the code review (2026-11-01).

Each test pins one bug that existed: a duplicate-ID citation counted as
supported, cache hits polluting latency percentiles, a stale index after a
chunker change, flattering eval metrics, and runs without a code version.
"""

import math
import subprocess
from pathlib import Path

import pytest

from docrag.answer.schema import Answer, AnswerResult, Citation, Evidence
from docrag.answer.verify import verify
from docrag.embed.base import EmbeddingService
from docrag.embed.fake import FakeEmbedder
from docrag.eval.dataset import EvalQuestion
from docrag.eval.judge import JudgeVerdict
from docrag.eval.runner import _evaluate, aggregate, question_row
from docrag.llm.fake import FakeProvider, default_responder
from docrag.llm.proxy import LLMProxy
from docrag.llm.types import Message
from docrag.rerank.reranker import FakeReranker
from docrag.shared import timing
from docrag.shared.cache import Cache
from docrag.shared.config import ExperimentConfig
from docrag.shared.provenance import code_version
from docrag.shared.schema import Chunk
from docrag.store.qdrant import QdrantStore, collection_name
from docrag.store.retriever import Retriever

# --- Verify: count citations, not IDs ------------------------------------


def test_same_id_with_real_and_invented_quote_is_partial() -> None:
    chunk = Chunk(
        id="d:p1:c0",
        doc_id="d",
        page_start=1,
        page_end=1,
        section_path=[],
        type="text",
        text="Revenue in Europe was 45.5 million euros.",
        embed_text="",
    )
    answer = Answer(
        answer="45.5 million [d:p1:c0]",
        evidence=[
            Evidence(id="d:p1:c0", quote="45.5 million euros"),  # real
            Evidence(id="d:p1:c0", quote="growth of 12 percent"),  # invented
        ],
    )
    result = verify("q", answer, [chunk])
    assert [c.verified for c in result.citations] == [True, False]
    assert result.status == "partially_supported"  # was "supported" before the fix


# --- Timing: cache hits get their own timer -------------------------------


@timing.timed("unit_cached")
def _maybe_cached(hit: bool) -> None:
    if hit:
        timing.cache_hit()


@timing.timed("unit_outer")
def _outer() -> None:
    _maybe_cached(hit=True)  # a cache hit INSIDE must not mark the outer call


def test_cache_hits_are_recorded_separately() -> None:
    timing.reset()
    _maybe_cached(hit=False)
    _maybe_cached(hit=True)
    _maybe_cached(hit=True)
    _outer()
    snap = timing.snapshot()
    assert snap["unit_cached"]["n"] == 1.0
    assert snap["unit_cached_cached"]["n"] == 3.0
    assert snap["unit_outer"]["n"] == 1.0
    assert "unit_outer_cached" not in snap


def test_embedding_and_llm_cache_hits(tmp_path: Path) -> None:
    import asyncio

    timing.reset()
    cache = Cache(tmp_path / "c.sqlite")
    service = EmbeddingService(FakeEmbedder(dim=8), cache=cache)
    service.embed(["a", "b"])
    service.embed(["a", "b"])  # all cached
    proxy = LLMProxy({"fake": FakeProvider()}, cache=cache)
    msgs = [Message(role="user", content="hi")]
    asyncio.run(proxy.chat("fake:m", msgs))
    asyncio.run(proxy.chat("fake:m", msgs))  # cached
    snap = timing.snapshot()
    assert snap["embed"]["n"] == 1.0 and snap["embed_cached"]["n"] == 1.0
    assert snap["llm"]["n"] == 1.0 and snap["llm_cached"]["n"] == 1.0


# --- Index fingerprint ----------------------------------------------------


def test_fingerprint_is_appended_to_the_collection_name() -> None:
    plain = collection_name("onnx:gte-small", 384, True)
    assert collection_name("onnx:gte-small", 384, True, fingerprint="ab12cd34") == (
        plain + "__vab12cd34"
    )


def test_chunker_version_bump_means_a_new_collection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from docrag import wiring
    from docrag.shared.settings import Settings

    settings = Settings(qdrant_location=":memory:", use_keyring=False)
    config = ExperimentConfig(name="t", embedder="fake:8", retrieval="hybrid")
    embedder = FakeEmbedder(dim=8)
    before = wiring.build_store(settings, config, embedder).collection
    monkeypatch.setattr(wiring, "CHUNKER_VERSION", "999")
    after = wiring.build_store(settings, config, embedder).collection
    assert before != after
    assert before.rsplit("__v", 1)[0] == after.rsplit("__v", 1)[0]  # same prefix


# --- Metrics: faithful only where the judge model read the answer ---------


def _q(answer: str = "45.5") -> EvalQuestion:
    return EvalQuestion(
        qid="q1",
        doc_id="tiny",
        question="Revenue?",
        answer=answer,
        answer_format="Str",
        evidence_pages=[2],
        slices=["table"],
    )


def _result(status: str, answer: str) -> AnswerResult:
    cits = (
        [Citation(id="tiny:p2:c1", quote="x", doc_id="tiny", page=2, verified=True)]
        if status == "supported"
        else []
    )
    return AnswerResult(
        question="Revenue?",
        answer=answer,
        raw_answer=answer,
        status=status,  # type: ignore[arg-type]
        citations=cits,
        context_ids=[],
    )


def test_abstention_is_not_counted_as_faithful() -> None:
    rule = JudgeVerdict(reasoning="rule: abstained", correct=False, faithful=True)
    row = question_row(_q(), [], _result("abstained", "Not answerable"), rule)
    assert row["correct"] is False
    assert math.isnan(row["faithful"])  # placeholder, not a measurement


def test_answer_to_unanswerable_question_is_not_counted_as_faithful() -> None:
    rule = JudgeVerdict(reasoning="rule: unanswerable", correct=False, faithful=True)
    row = question_row(
        _q("Not answerable"), [], _result("supported", "45.5 [tiny:p2:c1]"), rule
    )
    assert math.isnan(row["faithful"])


def test_judged_answer_keeps_faithful() -> None:
    llm = JudgeVerdict(reasoning="judge", correct=True, faithful=False)
    row = question_row(_q(), [], _result("supported", "45.5 [tiny:p2:c1]"), llm)
    assert row["faithful"] is False


async def test_failed_answers_count_as_wrong(
    memory_store: QdrantStore, embeddings: EmbeddingService
) -> None:
    def flaky(messages: list[Message], schema: type | None) -> str:
        if schema is not None and schema.__name__ == "Answer":
            if "Europe" in messages[-1].text:
                raise RuntimeError("provider down")
        return default_responder(messages, schema)

    config = ExperimentConfig(
        name="t",
        embedder="fake:64",
        reranker="fake:x",
        rerank_threshold=0.0,
        llm="fake:m",
        judge="fake:j",
    )
    retriever = Retriever(embeddings, memory_store, config, FakeReranker())
    proxy = LLMProxy({"fake": FakeProvider(flaky)})
    questions = [
        _q().model_copy(update={"qid": "ok", "question": "Who conducted the survey?"}),
        _q().model_copy(update={"qid": "bad", "question": "Revenue in Europe?"}),
    ]
    rows = await _evaluate(config, retriever, proxy, questions)
    by_qid = {r["qid"]: r for r in rows}
    assert by_qid["bad"]["failed"] is True and by_qid["bad"]["correct"] is False
    assert by_qid["ok"]["failed"] is False
    assert by_qid["ok"]["judge_failed"] is False
    metrics = aggregate(rows).filter(slice="all").row(0, named=True)
    assert metrics["failed"] == 0.5
    assert metrics["correct"] == 0.5  # before the fix: 1.0 (failure dropped)


# --- Code version in run.json ---------------------------------------------


def test_code_version_outside_git(tmp_path: Path) -> None:
    assert code_version(tmp_path) == {"commit": None, "dirty": None}


def test_code_version_commit_and_dirty(tmp_path: Path) -> None:
    def git(*args: str) -> None:
        subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True)

    git("init", "-q")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("x = 1\n")
    git("add", ".")
    git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "init")
    clean = code_version(tmp_path)
    assert clean["dirty"] is False and len(str(clean["commit"])) == 40

    (tmp_path / "README.md").write_text("docs only\n")  # not result-relevant
    assert code_version(tmp_path)["dirty"] is False
    (tmp_path / "src" / "a.py").write_text("x = 2\n")
    assert code_version(tmp_path)["dirty"] is True
