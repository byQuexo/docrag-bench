"""FastAPI routes. Only wiring: parse request -> call services -> shape response.

No business logic here, so the same AnswerService can be driven by the API, the
CLI and the eval runner, and tested without HTTP.

Run (from env + a config):  uv run docrag serve --config configs/baseline.yaml
"""

import asyncio
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from pydantic import BaseModel, Field

from docrag.answer.schema import AnswerResult
from docrag.answer.service import AnswerService
from docrag.ingest.pages import UnsafePathError, render_page_png, safe_pdf_path
from docrag.llm.proxy import BudgetExceededError, StructuredOutputError
from docrag.llm.types import Message, ProviderError
from docrag.shared import tracing
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


class QueryResponse(BaseModel):
    thread_id: str
    answer: str
    status: str
    sources: list[SourceOut]  # verified citations only
    rejected_citations: int
    latency_ms: float


def create_app(
    service: AnswerService,
    threads: ThreadStore | None = None,
    pdf_dir: Path | None = None,
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
                result, _ = await request.app.state.service.answer(
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
        return _to_response(thread_id, result, start)

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

    return app


def _to_response(thread_id: str, result: AnswerResult, start: float) -> QueryResponse:
    verified = result.verified_citations
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
        latency_ms=(time.perf_counter() - start) * 1000,
    )
