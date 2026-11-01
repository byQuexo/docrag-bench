"""The whole query pipeline with fakes: retrieve -> rerank -> LLM -> Verify."""

from docrag.answer.service import AnswerService
from docrag.embed.base import EmbeddingService
from docrag.llm.fake import FakeProvider
from docrag.llm.proxy import LLMProxy
from docrag.rerank.reranker import FakeReranker
from docrag.shared.config import ExperimentConfig
from docrag.store.qdrant import QdrantStore
from docrag.store.retriever import Retriever


def _service(
    store: QdrantStore,
    embeddings: EmbeddingService,
    provider: FakeProvider,
    **cfg: object,
) -> AnswerService:
    config = ExperimentConfig.model_validate(
        {"name": "t", "embedder": "fake:64", "reranker": "fake:x", **cfg}
    )
    retriever = Retriever(embeddings, store, config, FakeReranker())
    return AnswerService(retriever, LLMProxy({"fake": provider}), "fake:m")


async def test_answer_is_verified(
    memory_store: QdrantStore, embeddings: EmbeddingService
) -> None:
    provider = FakeProvider()
    service = _service(memory_store, embeddings, provider, rerank_threshold=0.3)
    result, hits = await service.answer(
        "What was the revenue in Europe in 2024?", doc_ids=["tiny"]
    )
    assert hits and hits[0].source == "rerank"
    assert result.status == "supported"
    assert result.citations and all(c.verified for c in result.citations)
    assert len(provider.calls) == 1
    # The context the LLM saw contains the chunk IDs as citation handles.
    system = provider.calls[0]["messages"][0]
    assert system.role == "system" and "[tiny:p2:c1]" in system.content


async def test_nothing_above_threshold_abstains_without_llm(
    memory_store: QdrantStore, embeddings: EmbeddingService
) -> None:
    provider = FakeProvider()
    service = _service(memory_store, embeddings, provider, rerank_threshold=0.99)
    result, _ = await service.answer("Who won the football world cup in 1954?")
    assert result.status == "abstained"
    assert result.abstained_before_llm
    assert provider.calls == []  # no tokens spent


async def test_history_goes_between_system_and_question(
    memory_store: QdrantStore, embeddings: EmbeddingService
) -> None:
    from docrag.llm.types import Message

    provider = FakeProvider()
    service = _service(memory_store, embeddings, provider, retrieval="hybrid")
    history = [
        Message(role="user", content="earlier q"),
        Message(role="assistant", content="earlier a"),
    ]
    await service.answer("revenue Europe", history=history)
    roles = [m.role for m in provider.calls[0]["messages"]]
    assert roles == ["system", "user", "assistant", "user"]


async def test_threshold_gates_the_answer_but_never_cuts_retrieval(
    memory_store: QdrantStore, embeddings: EmbeddingService
) -> None:
    """Regression test for the first full run: the threshold used to drop hits.

    With a threshold above every possible score (the fake reranker returns 0..1),
    retrieval must still return the ranked hits (recall stays measurable); only
    the ANSWER is skipped.
    """
    provider = FakeProvider()
    service = _service(memory_store, embeddings, provider, rerank_threshold=1.01)
    result, hits = await service.answer(
        "What was the revenue in Europe in 2024?", doc_ids=["tiny"]
    )
    assert hits, "ranking must not be cut by the threshold"
    assert result.abstained_before_llm and provider.calls == []
