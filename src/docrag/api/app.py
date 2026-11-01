"""FastAPI routes. Only wiring: parse request -> call services -> shape response.

No business logic here, so the same AnswerService can be driven by the API, the
CLI and the eval runner, and tested without HTTP.

Run: uv run docrag serve --config configs/answer_haiku_vlm_v3.yaml

If a built web UI exists (``ui/dist``, see ui/README.md), it is served at ``/``.
"""

import asyncio
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from docrag.answer.schema import AnswerResult
from docrag.answer.service import AnswerService
from docrag.eval.overview import overview, run_dirs
from docrag.ingest.pages import UnsafePathError, render_page_png, safe_pdf_path
from docrag.llm.proxy import BudgetExceededError, StructuredOutputError
from docrag.llm.types import Message, ProviderError
from docrag.shared import tracing
from docrag.shared.schema import Hit
from docrag.threads.store import InMemoryThreadStore, ThreadStore


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    thread_id: str | None = None
    doc_ids: list[str] | None = None  # restrict search to these documents


class SourceOut(BaseModel):
    id: str
    doc_id: str
    page: int
    quote: str
    page_image_url: str


class EvidenceOut(BaseModel):
    """One retrieved chunk as ranked after reranking (the UI's evidence list)."""

    rank: int
    id: str
    doc_id: str
    page: int
    type: str  # TEXT, TABLE, FIGURE or OCR (text read from a page image)
    score: float
    cited: bool  # backs a verified citation
    text: str  # first 300 characters


class QueryResponse(BaseModel):
    thread_id: str
    answer: str
    status: str
    sources: list[SourceOut]  # verified citations only
    rejected_citations: int
    evidence: list[EvidenceOut]
    abstained_before_llm: bool
    top_score: float | None
    threshold: float | None  # the abstain gate (only with reranking)
    input_tokens: int
    output_tokens: int
    cost_eur: float
    latency_ms: float


def create_app(
    service: AnswerService,
    threads: ThreadStore | None = None,
    pdf_dir: Path | None = None,
    results_dir: Path | None = None,
    ui_dir: Path | None = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # LEARN: code before `yield` runs at startup, after it at shutdown -
        # the modern replacement for @app.on_event("startup"/"shutdown").
        yield
        await service.proxy.aclose()

    # LEARN: FastAPI (0.162+) has built-in OpenTelemetry: once a global
    # TracerProvider exists (setup_tracing in `docrag serve`), every request gets
    # an HTTP server span "POST /query" (route, status code, and it continues a
    # caller's trace from the W3C `traceparent` header) plus endpoint and
    # serialization spans. Without a provider it stays off. We only exclude the
    # health check: a probe every few seconds would bury the real traces.
    app = FastAPI(
        title="docrag-bench",
        version="0.1.0",
        lifespan=lifespan,
        telemetry={"exclude": lambda scope: scope.get("path") == "/health"},
    )
    # app.state holds the "singletons" of this app instance. No module-level
    # globals, so two apps (e.g. in tests) never share state.
    app.state.service = service
    app.state.threads = threads or InMemoryThreadStore()
    app.state.pdf_dir = pdf_dir
    app.state.results_dir = results_dir
    app.state.documents = None  # computed on first request
    app.state.results_cache = (None, None)  # (results folder listing, overview)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    # LEARN: FastAPI reads the type hints: `body: QueryRequest` means "parse and
    # validate the JSON body as QueryRequest" (422 with details if invalid), and
    # `-> QueryResponse` becomes the documented response schema in /docs.
    @app.post("/query")
    async def query(body: QueryRequest, request: Request) -> QueryResponse:
        start = time.perf_counter()
        store: ThreadStore = request.app.state.threads
        thread_id = body.thread_id or store.create()
        history = store.get(thread_id)
        if history is None:
            raise HTTPException(404, f"unknown thread_id {thread_id!r}")
        # FastAPI's HTTP span is the trace root; "query" carries the RAG view of
        # the request. session.id = thread id on both: Phoenix groups all turns
        # of a conversation under "Sessions".
        tracing.annotate({"session.id": thread_id})
        with tracing.root_span(
            "query",
            attributes={
                "session.id": thread_id,
                "input.value": body.question,
                "docrag.history_turns": len(history) // 2,
            },
        ):
            try:
                result, hits = await request.app.state.service.answer(
                    body.question, doc_ids=body.doc_ids, history=history
                )
            except BudgetExceededError as err:
                raise HTTPException(402, str(err)) from err
            except (ProviderError, StructuredOutputError) as err:
                # 502 Bad Gateway: WE are fine, the upstream LLM misbehaved.
                raise HTTPException(502, str(err)) from err
            tracing.annotate(
                {"output.value": result.answer, "docrag.status": result.status}
            )
        store.append(
            thread_id,
            Message(role="user", content=body.question),
            Message(role="assistant", content=result.answer),
        )
        cfg = request.app.state.service.retriever.config
        gated = cfg.retrieval == "hybrid_rerank"
        return _to_response(
            thread_id, result, hits, cfg.rerank_threshold if gated else None, start
        )

    @app.get("/documents")
    async def documents(request: Request) -> list[str]:
        """Documents with a PDF on disk AND chunks in the served index."""
        if request.app.state.documents is None:
            base: Path | None = request.app.state.pdf_dir
            store = request.app.state.service.retriever.store

            def indexed() -> list[str]:
                if base is None:
                    return []
                pdfs = sorted(p.stem for p in base.glob("*.pdf"))
                return [d for d in pdfs if store.count(d) > 0]

            request.app.state.documents = await asyncio.to_thread(indexed)
        return request.app.state.documents  # type: ignore[no-any-return]

    @app.get("/results")
    async def results(request: Request) -> dict[str, Any]:
        """Dashboard data, computed from results/ (cached until a run is added)."""
        base: Path | None = request.app.state.results_dir
        if base is None:
            raise HTTPException(404, "results not configured")
        listing = tuple(d.name for d in run_dirs(base))
        cached_listing, cached = request.app.state.results_cache
        if cached_listing != listing:
            cached = await asyncio.to_thread(overview, base)
            request.app.state.results_cache = (listing, cached)
        return cached  # type: ignore[no-any-return]

    @app.get("/threads/{thread_id}")
    async def get_thread(thread_id: str, request: Request) -> list[Message]:
        msgs = request.app.state.threads.get(thread_id)
        if msgs is None:
            raise HTTPException(404, "unknown thread")
        return msgs  # type: ignore[no-any-return]

    @app.get("/pages/{doc_id}/{page}.png")
    async def page_image(doc_id: str, page: int, request: Request) -> Response:
        base: Path | None = request.app.state.pdf_dir
        if base is None:
            raise HTTPException(404, "page images not configured")
        try:
            path = safe_pdf_path(base, doc_id)
            # Rendering is CPU work (~50 ms): keep it off the event loop.
            png = await asyncio.to_thread(render_page_png, path, page)
        except UnsafePathError as err:
            raise HTTPException(400, str(err)) from err
        except (FileNotFoundError, IndexError) as err:
            raise HTTPException(404, str(err)) from err
        return Response(png, media_type="image/png")

    # Mounted LAST: the routes above win, every other path serves the UI.
    if ui_dir is not None and (ui_dir / "index.html").exists():
        app.mount("/", StaticFiles(directory=ui_dir, html=True), name="ui")
    return app


_TYPES = {"text": "TEXT", "table": "TABLE", "image": "FIGURE"}


def _to_response(
    thread_id: str,
    result: AnswerResult,
    hits: list[Hit],
    threshold: float | None,
    start: float,
) -> QueryResponse:
    verified = result.verified_citations
    cited = {c.id for c in verified}
    return QueryResponse(
        thread_id=thread_id,
        answer=result.answer,
        status=result.status,
        sources=[
            SourceOut(
                id=c.id,
                doc_id=c.doc_id,
                page=c.page,
                quote=c.quote,
                page_image_url=f"/pages/{c.doc_id}/{c.page}.png",
            )
            for c in verified
        ],
        rejected_citations=len(result.citations) - len(verified),
        evidence=[
            EvidenceOut(
                rank=rank,
                id=h.chunk.id,
                doc_id=h.chunk.doc_id,
                page=h.chunk.page_start,
                type="OCR" if h.chunk.origin != "text" else _TYPES[h.chunk.type],
                score=h.score,
                cited=h.chunk.id in cited,
                text=" ".join(h.chunk.text.split())[:300],
            )
            for rank, h in enumerate(hits, start=1)
        ],
        abstained_before_llm=result.abstained_before_llm,
        top_score=hits[0].score if hits else None,
        threshold=threshold,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        cost_eur=result.cost_eur,
        latency_ms=(time.perf_counter() - start) * 1000,
    )
