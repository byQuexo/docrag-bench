"""Retriever: question -> hits. Embed, search (dense/bm25/hybrid), optional rerank.

This is the exact code path used by the API AND the eval runner, so benchmark
numbers describe the system that is actually served.
"""

from dataclasses import dataclass

from docrag.embed.base import EmbeddingService
from docrag.rerank.reranker import Reranker, rerank
from docrag.shared import tracing
from docrag.shared.config import ExperimentConfig
from docrag.shared.schema import Hit
from docrag.shared.timing import timed
from docrag.store.qdrant import QdrantStore


@dataclass
class Retriever:
    embeddings: EmbeddingService
    store: QdrantStore
    config: ExperimentConfig
    reranker: Reranker | None = None

    def candidates(self, question: str, doc_ids: list[str] | None = None) -> list[Hit]:
        """First stage only (no rerank). With rerank, fetch more to rerank from."""
        cfg = self.config
        needs_vector = cfg.retrieval != "bm25"
        limit = cfg.prefetch_k if cfg.retrieval == "hybrid_rerank" else cfg.top_k
        return self.store.search(
            query_vector=self.embeddings.embed_query(question)
            if needs_vector
            else None,
            query_text=question,
            mode=cfg.retrieval,
            limit=limit,
            prefetch_k=cfg.prefetch_k,
            rrf_k=cfg.rrf_k,
            doc_ids=doc_ids,
        )

    # One span around the whole retrieval: its children are embed, search and
    # rerank, and it carries the FINAL ranked list - what the LLM will see.
    @timed("retrieve", kind=tracing.RETRIEVER)
    def retrieve(self, question: str, doc_ids: list[str] | None = None) -> list[Hit]:
        tracing.annotate(
            {"input.value": question, "docrag.retrieval": self.config.retrieval}
        )
        hits = self.candidates(question, doc_ids)
        if self.config.retrieval == "hybrid_rerank":
            if self.reranker is None:
                raise ValueError("retrieval=hybrid_rerank needs a reranker")
            # Rerank = reorder only. The threshold is an answer/abstain gate in
            # AnswerService, so retrieval metrics measure pure ranking quality.
            hits = rerank(self.reranker, question, hits, top_k=self.config.top_k)
        tracing.annotate_documents("retrieval.documents", tracing.hit_docs(hits))
        return hits
