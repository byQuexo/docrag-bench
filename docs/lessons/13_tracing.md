# Lesson 13: Tracing the question -> answer path (OpenTelemetry + Phoenix)

**Goal:** understand where tracing pays off in a RAG system and where it doesn't,
how one decorator became both a benchmark timer and a span, how the trace context
survives `await` and worker threads, and what to look at in Phoenix when an answer
is wrong.

Files: `src/docrag/shared/tracing.py`, `src/docrag/shared/timing.py`,
`src/docrag/answer/service.py`, `src/docrag/answer/verify.py`,
`src/docrag/store/retriever.py`, `src/docrag/store/qdrant.py`,
`src/docrag/rerank/reranker.py`, `src/docrag/embed/base.py`,
`src/docrag/llm/proxy.py`, `src/docrag/api/app.py`, `src/docrag/eval/runner.py`,
`src/docrag/cli.py`, `src/docrag/shared/settings.py`, `compose.yaml`,
`tests/conftest.py`, `tests/unit/test_tracing.py`.

---

## 1. Where tracing pays off, and where it doesn't

A trace is the story of **one request** told as a tree of timed steps (spans).
That is worth the most where one request crosses several stages and an external
service:

```
POST /query                         (FastAPI, HTTP status, route)
└── query                           (session.id = thread id)
    └── answer          CHAIN       question, final answer, status
        ├── retrieve    RETRIEVER   final ranked chunks
        │   ├── embed   EMBEDDING
        │   ├── search  RETRIEVER   50 first-stage candidates
        │   └── rerank  RERANKER    top-k after the cross-encoder
        ├── llm         LLM         prompt messages, reply, tokens, EUR
        └── verify      GUARDRAIL   status, which citations were rejected and why
```

In a RAG system, latency is only half of the value. The other half is **debugging
quality**: a wrong answer was caused either by retrieval (the right chunk wasn't in
the context) or by the LLM (it was there and the model ignored it). With the chunk
IDs on `retrieve`, the prompt on `llm` and the rejection reasons on `verify`, one
click in Phoenix answers that question for a single request.

**Ingest is deliberately not traced.** It's an offline batch job: nobody waits on
one page, and a span per page or chunk would flood the trace store with thousands
of spans that nobody reads. `@timed` already gives p50/p95 for parse/embed/upsert,
and the logs say which documents failed.

| Entry point | Tracing | Why |
|---|---|---|
| `docrag serve` (API) | yes, one trace per request | user-facing latency, debugging single answers |
| `docrag eval` | yes, one trace per question (+ judge) | error analysis: filter by run, slice, status |
| `docrag ask` | yes, one trace | the same pipeline from the command line |
| `docrag ingest` | no | batch job, `@timed` + logs are enough |

---

## 2. Two packages: API and SDK

OpenTelemetry splits into the **API** (`opentelemetry-api`: `trace.get_tracer`,
`start_as_current_span`, ...) and the **SDK** (`opentelemetry-sdk`: the
`TracerProvider` that actually records spans and hands them to exporters).

Library code only talks to the API. As long as nobody installs an SDK provider,
every API call goes to a no-op implementation. That's what makes "ingest has no
tracing" free: the code is the same, the process just never calls
`setup_tracing()`.

> Java analogy: the API is SLF4J, the SDK is Logback. Libraries log against SLF4J;
> the application decides whether anything gets written and where.

```python
# shared/tracing.py
def setup_tracing(endpoint, service_name="docrag", exporter=None) -> bool:
    if endpoint is None and exporter is None:
        return False  # tracing stays a no-op
    from opentelemetry.sdk.trace import TracerProvider  # lazy: SDK only if used

    ...
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint)))
    trace.set_tracer_provider(provider)
```

`BatchSpanProcessor` buffers finished spans and exports them from a background
thread. A slow or missing Phoenix therefore never adds latency to a request; at
worst spans get dropped. The provider flushes the buffer at interpreter exit, so a
CLI run (`docrag ask`) doesn't lose its last spans.

The endpoint comes from `DOCRAG_OTEL_ENDPOINT` (`shared/settings.py`). Unset = off.
Only `serve`, `ask` and `eval` call `setup_tracing`; `ingest` never does.

---

## 3. The decorator that became a span

Lesson 01 promised: "Timing decorator before OTel ... moving to spans later only
changes `timing.py`." Here is that change:

```python
# shared/timing.py
@contextmanager
def _span(name: str, kind: str | None, root: bool) -> Iterator[None]:
    if not (root or tracing.in_trace()):
        yield  # outside a trace: no span at all
        return
    attrs = {tracing.SPAN_KIND: kind} if kind else None
    with tracing.tracer().start_as_current_span(name, attributes=attrs):
        yield


def timed(name, *, kind=None, root=False):
    def decorator(func):
        ...

        def wrapper(*args, **kwargs):
            with _span(name, kind, root):
                start = time.perf_counter()
                try:
                    return func(*args, **kwargs)
                finally:
                    record(name, time.perf_counter() - start)
```

The call sites only gained a `kind=` (the OpenInference span kind, section 6):

```python
@timed("rerank", kind=tracing.RERANKER)       # rerank/reranker.py
@timed("llm", kind=tracing.LLM)               # llm/proxy.py
@timed("answer", kind=tracing.CHAIN, root=True)   # answer/service.py
```

### "Only inside a trace" is the key rule

`embed()` runs both during ingest (thousands of times) and during a query (once).
The decorator can't know which one is happening. The **context** can: during a
query there is an active span, during ingest there isn't. So `@timed` only *joins*
traces and never *starts* one. Only explicit entry points start traces:

* `AnswerService.answer` has `root=True`, so `docrag ask` gets a trace too;
* the API: FastAPI's own HTTP span (section 5);
* the eval runner: one `root_span("eval.question")` per question.

This also covers ingest *inside* `docrag eval`: `run_experiment` indexes missing
documents before answering. Tracing is on in that process, but indexing runs
outside any question span, so it records nothing. The smoke test in section 8
checks exactly that.

The benchmark registry (`record(...)`) still runs in every case, so `run.json`
latency numbers don't depend on whether tracing is on.

---

## 4. How the context survives `await` and threads

"Current span" is stored in a **`contextvars.ContextVar`**. A contextvar is like
a thread-local, but it also follows asyncio: every `asyncio.Task` gets a copy of
the context at the moment it is created.

```python
# answer/service.py
hits = await asyncio.to_thread(self.retriever.retrieve, question, doc_ids)
```

`retrieve` runs in a worker thread, yet its span is a child of `answer`. That
works because **`asyncio.to_thread` copies the current context into the thread**
(it calls `contextvars.copy_context().run(...)`). A plain
`ThreadPoolExecutor.submit(fn)` does *not*; the span would become an orphan root.
`test_answer_is_one_connected_trace` checks the parent link explicitly.

Same for concurrency in the eval: `bounded_gather` -> `asyncio.gather` wraps each
coroutine in a Task, each Task gets its own context copy, so 4 questions running
at the same time build 4 separate trees without interfering.

> TS analogy: `AsyncLocalStorage` in Node. Java: a `ThreadLocal` that you'd have
> to propagate by hand to executors (or OTel's `Context.taskWrapping`). Rust:
> `tracing`'s `Instrument` trait, which attaches a span to a future.

And multiprocessing? The PDF parser runs in worker *processes*. Contextvars don't
cross process boundaries at all, which is fine: that's ingest, untraced by design.

---

## 5. The roots: API, eval, and the judge

### API: FastAPI traces itself

FastAPI 0.162+ has built-in OpenTelemetry. Once a global `TracerProvider` exists,
every request gets an HTTP server span (`POST /query`, route, status code) and
FastAPI also **continues a caller's trace** from the W3C `traceparent` header. So
if a frontend or gateway is traced too, its spans and ours end up in one trace.

We add our own `query` span below it, with `session.id = thread_id`. Phoenix
groups all traces that share a `session.id` into one conversation ("Sessions"
tab), which is exactly what a thread is.

```python
# api/app.py
app = FastAPI(..., telemetry={"exclude": lambda scope: scope.get("path") == "/health"})
...
tracing.annotate({"session.id": thread_id})          # on FastAPI's HTTP span
with tracing.root_span("query", attributes={"session.id": thread_id, ...}):
    result, _ = await request.app.state.service.answer(...)
```

The health check is excluded: a container probe every 10 seconds would produce
8,640 useless traces a day.

### Eval: one trace per question, not per run

A run with 735 questions as one trace would be one giant unreadable tree. Instead,
each question is its own trace with filterable attributes (`docrag.run`,
`docrag.config_hash`, `docrag.qid`, `docrag.slices`, `docrag.answerable`). In
Phoenix you can then filter, for example, "run `answer_haiku`, slice `table`,
`docrag.status == unsupported`".

### The judge joins a trace that has already ended

The runner answers *all* questions first and judges them afterwards (a different
concurrency limit, and a judge failure must not lose answers). By then the
question span is closed. OpenTelemetry doesn't mind: a parent only needs its
**SpanContext** (trace id + span id), not a live span object.

```python
# eval/runner.py
with tracing.root_span("eval.question", attributes=attrs(q)) as span:
    question_spans[q.qid] = span.get_span_context()  # remember the IDs
    return await service.answer(...)
...
with tracing.child_of(question_spans.get(q.qid), "judge", kind=tracing.EVALUATOR):
    verdict = await judge(...)
```

`child_of` wraps the stored context in a `NonRecordingSpan` and starts the new
span under it. In Phoenix the judge appears inside the question's trace, with
`docrag.correct`, `docrag.faithful` and the judge's reasoning.

---

## 6. Attributes: OpenInference conventions

Span names and timings alone don't explain a wrong answer. The attributes do.
We use the **OpenInference** names (Arize's semantic conventions for LLM apps)
because Phoenix renders them specially:

| Span | Attributes | What Phoenix shows |
|---|---|---|
| `llm` | `llm.input_messages.{i}.message.role/content`, `llm.output_messages.0...`, `llm.token_count.prompt/completion/total`, `llm.model_name`, `llm.provider` | chat view of the prompt, token counts |
| `retrieve`, `search` | `retrieval.documents.{i}.document.id/score/content` | ranked document list |
| `rerank` | `reranker.output_documents.{i}...`, `reranker.model_name`, `reranker.top_k` | reranked list |
| `embed` | `embedding.model_name` | |
| all | `openinference.span.kind`, `input.value`, `output.value` | icon, input/output panes |

Project-specific attributes use a `docrag.` prefix: `docrag.cost_eur`,
`docrag.cache_hit`, `docrag.top_score`, `docrag.rerank_threshold`,
`docrag.verify.rejected` and so on.

`tracing.annotate({...})` sets attributes on whatever span is current. It returns
immediately when that span isn't recording, so domain code calls it
unconditionally without `if tracing_enabled:` branches. Values over 8,000 chars
are truncated (a span is not a log archive), and images are dropped from LLM
messages (`m.text`), since base64 page renders don't belong in a span.

---

## 7. Running it

```powershell
docker compose up -d qdrant phoenix
$env:DOCRAG_OTEL_ENDPOINT = "http://127.0.0.1:6006/v1/traces"
uv run docrag ask --config configs/baseline.yaml --doc PH_2016.06.08_Economy-Final "How many Latinos were surveyed?"
# open http://127.0.0.1:6006 -> project "default" -> the trace
```

Or the whole stack: `docker compose up -d` (the `api` service already has
`DOCRAG_OTEL_ENDPOINT` pointing at `phoenix`), then send a request via
http://127.0.0.1:8000/docs.

Things to look at:

1. **An "unsupported" answer:** open `verify` -> `docrag.verify.rejected` says
   which citation failed and why. Then `llm` -> was the cited chunk in the prompt?
2. **An abstention:** `answer` -> `docrag.top_score` vs `docrag.rerank_threshold`.
   Then `rerank` -> was the gold page in the candidates at all?
3. **A slow request:** the waterfall. On CPU, `rerank` dominates (lesson 10, D2.1).

---

## 8. Testing it

`tests/conftest.py` installs a provider with an `InMemorySpanExporter` once per
test session (the global provider can only be set once per process). The `spans`
fixture clears it per test. Tests that don't ask for `spans` are unaffected.

`tests/unit/test_tracing.py` checks the design rules, not OTel itself:

* `@timed` outside a trace: duration recorded, no span;
* ingest (`ingest_pdf` + `embed`): no spans;
* one `answer()` call: one trace, all 7 spans, `retrieve` parented across the
  thread hop, LLM tokens + retrieved document IDs present;
* API: FastAPI's span is the root, `query` carries `session.id`, `/health` is not traced;
* `child_of` joins the trace of an already-ended span.

The OTLP exporter itself was smoke-tested against a tiny local HTTP server that
decodes the protobuf payload: exactly 1 trace with the 7 query spans arrived, and
the ingest work in the same process produced none.

---

## 9. Known limitations

* **Traces contain content**: questions, prompts, document text. Fine for a local
  Phoenix bound to 127.0.0.1; in production you'd redact or switch content capture
  off (OpenInference has `OPENINFERENCE_HIDE_INPUTS`-style switches) and set
  retention.
* **VLM-OCR calls during ingest are not traced** (no root). Their cost shows up in
  the proxy stats, not in Phoenix. That follows from the rule in section 3; adding
  `root=True` to the LLM span would trace them, at the cost of one trace per OCR'd
  page batch.
* **No sampling.** Every request is traced. At real traffic you'd sample
  (`ParentBased(TraceIdRatioBased(0.1))`) or tail-sample in a collector, keeping
  all errors.
* **Cached LLM calls still create `llm` spans** (with `docrag.cache_hit = true` and
  0 EUR). That's intended: the span shows what the call would cost on a cold
  cache, and why a request was fast.
* `BatchSpanProcessor` drops spans if the process is killed hard (`kill -9`) before
  a flush.

---

## Python concepts in this lesson

* **`contextvars`**: per-task (and per-thread) "current value" storage. asyncio
  copies the context into every Task; `asyncio.to_thread` copies it into the worker
  thread; `ThreadPoolExecutor.submit` does not.
* **`@contextmanager` generators**: code before `yield` = enter, after = exit. The
  `_span` helper yields once either way, with or without a real span around it.
* **Context managers stack inside a decorator**: `with _span(...): try/finally`
  means the span also covers the time measurement and ends even on exceptions.
  OTel records the exception and sets ERROR status on the span automatically.
* **Lazy imports for optional runtime pieces**: the SDK and exporter are imported
  inside `setup_tracing`, so code paths without tracing never load them.
* **Closures over loop state**: `question_spans` is a dict in `_evaluate` that the
  nested `answer()` and `judge_in_trace()` coroutines both use.

## Exercises

1. **See the thread hop break.** In `answer/service.py`, replace
   `asyncio.to_thread(...)` with
   `loop.run_in_executor(None, self.retriever.retrieve, question, doc_ids)`
   (`loop = asyncio.get_running_loop()`). Run `uv run pytest tests/unit/test_tracing.py`.
   *Done when* you can explain which assertion fails and why. Revert afterwards.
2. **Return the trace ID.** Add an `X-Trace-Id` response header in `/query`
   (`format(span.get_span_context().trace_id, "032x")`). *Done when* a test reads
   the header and finds a span with that trace ID in `spans`.
3. **Sampling.** Add `DOCRAG_OTEL_SAMPLE_RATIO` (default 1.0) and pass
   `ParentBased(TraceIdRatioBased(ratio))` to the `TracerProvider`. *Done when* a
   test with ratio 0.0 records no spans.
4. **Hide content.** Add a setting that makes `annotate` drop `input.value`,
   `output.value`, `llm.*_messages.*` and `*.document.content`. *Done when* a test
   proves no question text reaches the exporter with the flag on.
5. **Trace a real eval run.** Start Phoenix, run `docrag eval configs/baseline.yaml`
   with the endpoint set, and find 3 questions with `docrag.status = unsupported`.
   *Done when* you can say for each one whether retrieval or the LLM caused it.

## Interview questions

**Q: Where did you add tracing, and where deliberately not?**
On the query path (API request, eval question, CLI ask), because one request
crosses embed, search, rerank, an external LLM and verify, and a trace shows both
the latency breakdown and the data (chunks, prompt, rejected citations) needed to
debug a single wrong answer. Not on ingest: it's an offline batch job, nobody
waits on a page, and per-chunk spans would flood the trace store. `@timed`
percentiles and logs cover it.

**Q: The same `embed()` runs in ingest and in queries. How does it know whether to create a span?**
It doesn't decide; the context does. `@timed` only joins an existing trace (an
active span in the contextvar) and never starts one. Traces start at explicit
roots. During ingest there's no root, so the decorator costs a contextvar lookup
and records nothing.

**Q: How does the trace context survive `await` and threads?**
OTel stores the current span in a `contextvars.ContextVar`. asyncio copies the
context into every Task, and `asyncio.to_thread` copies it into the worker thread,
so the `retrieve` span in the thread still knows its parent. A plain executor
`submit` doesn't copy it; I have a test that checks the parent link across the
thread hop.

**Q: Why OpenTelemetry instead of a vendor SDK like LangSmith or Langfuse?**
OTel is the vendor-neutral standard: the code depends on the API only, and the
backend is a config value. Phoenix reads OTLP plus OpenInference attributes, but
Jaeger, Tempo, Datadog or Langfuse (which also ingests OTLP) would receive the same
spans. FastAPI emits OTel natively, so HTTP spans and RAG spans land in one trace
without glue code.

**Q: What would you change for production?**
Sampling (parent-based ratio or tail-based in an OTel Collector, keeping all
errors), content redaction or capture switched off for PII, an OTel Collector
between the app and the backend for batching/retries/routing, and metrics
(request rate, p95, abstention rate, rejected-citation rate) next to traces,
since traces are for single requests and metrics for trends and alerts.

**Q: Why keep the old `@timed` registry if you have spans now?**
They answer different questions. The registry gives p50/p95 per stage for a
benchmark run, in `run.json`, with zero infrastructure and reproducibly in CI.
Spans explain *one* request and need a running backend. The benchmark results
must not depend on whether Phoenix happens to be up.
