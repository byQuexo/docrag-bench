"""AnswerService: question (+ thread history) -> verified AnswerResult.

retrieve -> (abstain if nothing relevant) -> render context -> LLM (structured)
-> Verify.
"""

import asyncio
from dataclasses import dataclass

from docrag.answer.context import load_prompt, render_context
from docrag.answer.schema import NOT_ANSWERABLE, Answer, AnswerResult
from docrag.answer.verify import verify
from docrag.llm.proxy import LLMProxy
from docrag.llm.types import Message
from docrag.shared import tracing
from docrag.shared.schema import Hit
from docrag.shared.timing import timed
from docrag.store.retriever import Retriever


@dataclass
class AnswerService:
    retriever: Retriever
    proxy: LLMProxy
    llm_route: str

    # root=True: `docrag ask` calls this directly and still gets a full trace.
    # Under the API or eval it becomes a child of their own root span.
    @timed("answer", kind=tracing.CHAIN, root=True)
    async def answer(
        self,
        question: str,
        *,
        doc_ids: list[str] | None = None,
        history: list[Message] | None = None,
    ) -> tuple[AnswerResult, list[Hit]]:
        # LEARN: embedding/reranking is CPU-bound, synchronous code. Calling it
        # directly inside an `async def` would block the event loop, so every other
        # request in the FastAPI server would wait. asyncio.to_thread runs it in a
        # worker thread (ONNX Runtime releases the GIL while computing).
        tracing.annotate({"input.value": question, "docrag.llm_route": self.llm_route})
        # LEARN: asyncio.to_thread copies the current contextvars into the worker
        # thread, so the "retrieve" span opened over there still knows its parent.
        # A plain ThreadPoolExecutor.submit() would NOT do that (orphaned spans).
        hits = await asyncio.to_thread(self.retriever.retrieve, question, doc_ids)
        cfg = self.retriever.config
        context_hits = hits[: cfg.context_k]
        # The abstention gate: with a reranker, its scores are calibrated
        # relevance estimates, so "even the best chunk is below the threshold"
        # means "the document probably doesn't contain the answer".
        gated = cfg.retrieval == "hybrid_rerank" and bool(hits)
        if gated and hits[0].score < cfg.rerank_threshold:
            context_hits = []
        tracing.annotate(
            {
                "docrag.top_score": hits[0].score if hits else None,
                "docrag.rerank_threshold": cfg.rerank_threshold if gated else None,
                "docrag.n_context": len(context_hits),
            }
        )

        if not context_hits:
            # Nothing relevant enough: abstain without spending an LLM call.
            # This is the "honest" path for unanswerable questions.
            result = AnswerResult(
                question=question,
                answer=NOT_ANSWERABLE,
                raw_answer=NOT_ANSWERABLE,
                status="abstained",
                citations=[],
                context_ids=[],
                abstained_before_llm=True,
            )
            tracing.annotate(
                {
                    "output.value": result.answer,
                    "docrag.status": result.status,
                    "docrag.abstained_before_llm": True,
                }
            )
            return result, hits

        prompt = load_prompt(self.retriever.config.prompt_version)
        # Static instructions first, then the context: providers cache the longest
        # identical PREFIX of a prompt, so the unchanging part must come first.
        messages = [
            Message(role="system", content=f"{prompt}\n{render_context(context_hits)}"),
            *(history or []),
            Message(role="user", content=question),
        ]
        parsed, _ = await self.proxy.structured(
            self.llm_route,
            messages,
            Answer,
            temperature=self.retriever.config.llm_temperature,
        )
        result = verify(question, parsed, [h.chunk for h in context_hits])
        tracing.annotate(
            {"output.value": result.answer, "docrag.status": result.status}
        )
        return result, hits
