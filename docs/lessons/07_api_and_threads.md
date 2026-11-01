# Lesson 07: API and threads (`api/app.py`, `threads/store.py`, `wiring.py`)

**Goal:** understand how the services are wired together and exposed over HTTP with
FastAPI, how conversation state is kept out of the API, and how CPU work is kept off
the event loop. Milestone M4: "FastAPI /query, threads, E2E test in CI".

## What happens step by step

### 1. The composition root (`wiring.py`)

```python
def build_retriever(settings, config, cache) -> Retriever:
    embedder = make_embedder(config.embedder, settings.models_dir)
    embeddings = EmbeddingService(embedder, cache, settings.embed_batch_size)
    reranker = (
        make_reranker(config.reranker, settings.models_dir)
        if config.retrieval == "hybrid_rerank"
        else None
    )
    return Retriever(
        embeddings, build_store(settings, config, embedder.dim), config, reranker
    )


def build_answer_service(settings, config, cache) -> AnswerService:
    ...
    return AnswerService(
        build_retriever(settings, config, cache),
        LLMProxy.from_settings(settings, cache),
        config.llm,
    )
```

Every class receives its dependencies as constructor arguments; only `wiring.py`
knows how to build the real graph. Tests build the same graph with
`FakeEmbedder`, `FakeReranker`, `FakeProvider` and `QdrantClient(":memory:")`. This is
dependency injection without a framework (Spring/NestJS do the same via annotations).

### 2. The app factory (`api/app.py`)

```python
def create_app(
    service: AnswerService,
    threads: ThreadStore | None = None,
    pdf_dir: Path | None = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        yield
        await service.proxy.aclose()

    app = FastAPI(title="docrag-bench", version="0.1.0", lifespan=lifespan)
    app.state.service = service
    app.state.threads = threads or InMemoryThreadStore()
    app.state.pdf_dir = pdf_dir
    ...
    return app
```

- **Factory, not a module-level `app`:** two apps (e.g. two tests) never share state.
- **`lifespan`:** code before `yield` = startup, after = shutdown (closes HTTP
  connection pools).
- **`app.state`:** the per-app "singletons". Handlers read them via `request.app.state`.

`docrag serve --config ...` builds the service from settings + config and calls
`uvicorn.run(create_app(...))`.

### 3. `POST /query`

```python
class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    thread_id: str | None = None
    doc_ids: list[str] | None = None
```

1. FastAPI validates the JSON body against `QueryRequest`; errors → **422** with
   details, automatically.
2. No `thread_id` → create a thread. Unknown id → **404**.
3. `service.answer(question, doc_ids=..., history=history)`.
4. Error mapping:
   - `BudgetExceededError` → **402** Payment Required
   - `ProviderError` / `StructuredOutputError` → **502** Bad Gateway ("we are fine, the
     upstream LLM misbehaved")
5. Append the user + assistant messages to the thread.
6. Respond with **verified sources only**, plus `rejected_citations` count, `status`,
   `latency_ms`, and a `page_image_url` per source.

Example response:

```json
{
  "thread_id": "3f2c...",
  "answer": "Europe had 45.5 million EUR revenue in 2024 [tiny:p2:c1].",
  "status": "supported",
  "sources": [{"id": "tiny:p2:c1", "doc_id": "tiny", "page": 2,
               "quote": "Europe 41.0 45.5", "page_image_url": "/pages/tiny/2.png"}],
  "rejected_citations": 0,
  "latency_ms": 812.4
}
```

### 4. `GET /pages/{doc_id}/{page}.png`

```python
path = safe_pdf_path(base, doc_id)
png = await asyncio.to_thread(render_page_png, path, page)
```

`doc_id` comes from the URL, so it's untrusted. `safe_pdf_path` validates it, resolves the
path and checks `is_relative_to(base)` → **400** on traversal attempts. Bad page →
**404**. Rendering runs in a thread.

### 5. Threads (`threads/store.py`)

```python
class ThreadStore(Protocol):
    def create(self) -> str: ...
    def get(self, thread_id: str) -> list[Message] | None: ...
    def append(self, thread_id: str, *messages: Message) -> None: ...
```

`InMemoryThreadStore` is a dict guarded by a `threading.Lock`; `get` returns a
**copy**; history is trimmed to the last 20 messages. The API itself is stateless:
once the store is Redis-backed, any API replica can serve any request.

Note that retrieval uses only the current question; the history only goes to the LLM.
A follow-up like "and in 2023?" therefore retrieves poorly. That's a known limitation,
and the fix is query rewriting from history (future work).

### 6. `asyncio.to_thread` for CPU work

FastAPI runs `async def` handlers on one event loop. A 300 ms ONNX call directly inside
the handler would freeze **all** requests for 300 ms. `AnswerService.answer` does
`await asyncio.to_thread(self.retriever.retrieve, ...)`: the work runs in a worker
thread, and ONNX Runtime and numpy release the GIL while computing, so other requests
continue.

Try it: `uv run docrag serve --config configs/baseline.yaml` and open
`http://127.0.0.1:8000/docs`. The OpenAPI UI is generated from the type hints.

## Why this design

> - **Routes are only wiring:** the same AnswerService is used by API, CLI and eval,
>   so the benchmark measures the served system.
> - **Stateless API + ThreadStore interface:** horizontal scaling becomes a
>   configuration change (Redis), not a rewrite.
> - **Precise status codes:** clients (and monitoring) can tell "your request is
>   bad" (4xx) from "the LLM provider failed" (502) from "budget exhausted" (402).

## Python concepts

### FastAPI vs Express/Spring/axum

```python
@app.post("/query")
async def query(body: QueryRequest, request: Request) -> QueryResponse:
```

Type hints drive everything: body parsing + validation (Pydantic), the response schema,
and OpenAPI docs. In Express you'd validate manually (or with zod); Spring does it with
`@RequestBody @Valid`; axum uses typed extractors (`Json<QueryRequest>`).

### `@asynccontextmanager`

```python
from contextlib import asynccontextmanager


@asynccontextmanager
async def resource():
    print("open")
    try:
        yield "handle"
    finally:
        print("close")
```

A generator function becomes a context manager: setup before `yield`, teardown after.

### `*messages` in a signature

`def append(self, thread_id: str, *messages: Message)` takes any number of messages,
like `...messages: Message[]` in TS or `Message... messages` in Java.

### Locks

`with self._lock:` = Java `synchronized` block / Rust `Mutex::lock()` guard. Needed
because FastAPI may call sync code from worker threads.

## Exercises

1. Start the server with `configs/e2e_fake.yaml` (fake models, `DOCRAG_QDRANT_LOCATION=:memory:`
   after ingesting fixtures in a test) or use `TestClient` in a test, and POST two
   questions with the same `thread_id`. **Done when** `GET /threads/{id}` shows 4 messages.
2. Request `/pages/..%2F..%2Fsecret/1.png`. **Done when** you get 400 or 404 and know
   which check stopped it.
3. Replace `asyncio.to_thread(...)` with a direct call in `AnswerService.answer`, fire 5
   concurrent requests (e.g. `httpx.AsyncClient` + `asyncio.gather`) and compare total
   time. Revert. **Done when** you've measured the difference.
4. Add `GET /config` that returns the current `ExperimentConfig` (no secrets).
   **Done when** it shows in `/docs`.

## Interview questions

**Q: How is the API structured?**
A: An app factory `create_app(service, threads, pdf_dir)`; routes only parse, call the
AnswerService and shape the response. Dependencies are passed in and stored on
`app.state`, so tests build the app with fakes.

**Q: What happens if two users write to the same thread at the same time?**
A: The in-memory store is lock-protected, so no data is corrupted, but both requests read
the history before either appends, so the two Q/A pairs land in arrival order and neither
answer saw the other. For strict ordering you'd add a per-thread lock or optimistic
versioning in the store.

**Q: How would you scale the API?**
A: It's stateless apart from the ThreadStore, so: move threads and the cache to Redis,
run N uvicorn workers/replicas behind a load balancer, run Qdrant as a cluster, and move
embedding/reranking to a GPU inference service if CPU becomes the bottleneck.

**Q: What happens if Qdrant is down?**
A: The search call raises a connection error in the retriever. It isn't one of the mapped
exceptions, so FastAPI returns 500. A production version would map it to 503 with a
retry-after header and expose a readiness probe that checks Qdrant (`/health` is liveness only).

**Q: Why `asyncio.to_thread` for retrieval?**
A: Embedding and reranking are synchronous CPU work. Inside an async handler they'd block
the single event loop and every concurrent request would wait. In a thread they run in
parallel with other requests, because ONNX Runtime releases the GIL.

**Q: How do you protect the page image endpoint?**
A: The doc_id is validated against a strict regex, the path is resolved and checked to be
inside the PDF directory with `is_relative_to`. That blocks `../` traversal and symlink
escapes.

**Q: What happens with 1000 concurrent requests?**
A: Each needs a reranker pass (~1 s CPU) and an LLM call. The default thread pool and the
LLM provider's rate limits become the bottleneck, and latency grows. Fixes: bound concurrency
with a semaphore and return 429 early, batch reranking on a GPU service, cache answers, and
scale replicas.
