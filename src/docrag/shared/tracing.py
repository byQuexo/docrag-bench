"""OpenTelemetry tracing for the question -> answer path, viewed in Phoenix.

Design (see docs/lessons/13_tracing.md):

* Every ``@timed`` function also opens a span (shared/timing.py). Call sites
  didn't change when tracing was added.
* A span is only recorded INSIDE a trace. Traces start at explicit roots: an API
  request, one eval question, ``docrag ask``. Ingest (parse, embed, upsert) has no
  root, so it produces no spans, even when it runs inside ``docrag eval``.
* Exporting is opt-in: ``setup_tracing()`` is only called by serve/ask/eval and
  only when ``DOCRAG_OTEL_ENDPOINT`` (Phoenix) and/or the Langfuse keys are set.
  Without them the OpenTelemetry API is a no-op (one function call, no
  allocation worth measuring). ``OTEL_SDK_DISABLED=true`` (the standard OTel
  switch) turns everything off even when they are set.
* Langfuse is a second OTLP exporter on the same spans, not a second SDK: one
  instrumentation, two viewers. Langfuse reads ``langfuse.observation.type`` and
  the ``gen_ai.*`` conventions, so both are set next to the OpenInference keys.

Attribute names follow OpenInference (https://github.com/Arize-ai/openinference),
the semantic conventions Phoenix understands: it shows LLM spans with tokens and
the prompt, retriever spans with a document list, and so on.
"""

import base64
import json
import logging
import os
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

from opentelemetry import trace
from opentelemetry.context import Context
from opentelemetry.trace import Span, SpanContext

log = logging.getLogger(__name__)

TRACER_NAME = "docrag"
# OpenInference span kinds. Phoenix uses them for icons and specialised views.
CHAIN = "CHAIN"
RETRIEVER = "RETRIEVER"
RERANKER = "RERANKER"
EMBEDDING = "EMBEDDING"
LLM = "LLM"
EVALUATOR = "EVALUATOR"
GUARDRAIL = "GUARDRAIL"  # checks an output, e.g. our citation Verify
SPAN_KIND = "openinference.span.kind"
# The same idea in Langfuse's vocabulary ("generation" = an LLM call, which
# Langfuse shows with model, tokens and cost).
LANGFUSE_TYPE = "langfuse.observation.type"
_LANGFUSE_TYPES = {
    CHAIN: "chain",
    RETRIEVER: "retriever",
    RERANKER: "retriever",
    EMBEDDING: "embedding",
    LLM: "generation",
    EVALUATOR: "evaluator",
    GUARDRAIL: "guardrail",
}

# Long prompts/contexts are truncated: a span is not a log archive, and Phoenix
# gets slow with megabyte-sized attributes.
MAX_TEXT = 8_000
# The LLM input (system prompt with up to 10 retrieved chunks) is the one value
# worth keeping whole: it is exactly what the model saw when it answered.
MAX_PROMPT_TEXT = 64_000
# OpenTelemetry keeps 128 attributes per span by default and silently drops the
# OLDEST ones beyond that - which are the span-kind keys set at span start. The
# search span alone carries 50 candidates x 3 attributes.
MAX_SPAN_ATTRIBUTES = 512

AttrValue = str | bool | int | float | list[str]

# Root span name -> Langfuse trace name. Verb-first and stable, because saved
# filters, dashboards and evaluators match on trace names. (Span names stay as
# they are: they double as the latency keys in run.json.)
# (FastAPI's HTTP span is called just "POST" when it starts, so the /query
# handler names its trace itself.)
TRACE_NAMES = {
    "answer": "answer-question",
    "eval.question": "eval-question",
}

# Attributes that hold user or document TEXT (as opposed to ids, scores, counts,
# tokens, cost). Dropped when content capture is off (DOCRAG_TRACE_CONTENT=false).
_CONTENT_KEYS = frozenset(
    {
        "input.value",
        "output.value",
        "langfuse.observation.input",
        "langfuse.observation.output",
        "reranker.query",
        "docrag.verify.rejected",  # contains reasons quoting the answer
    }
)


def is_content(key: str) -> bool:
    # *.content covers llm.input_messages.N.message.content and
    # retrieval.documents.N.document.content.
    return key in _CONTENT_KEYS or key.endswith(".content")


@dataclass(frozen=True)
class TraceContext:
    """Per-process trace settings, applied to every span by the span processor."""

    environment: str | None = None  # development / ci / production
    release: str | None = None  # e.g. image tag or git sha
    tags: tuple[str, ...] = ()  # e.g. ("api",): which entry point made the trace
    capture_content: bool = True  # False = no questions, prompts or document text


_CONTEXT = TraceContext()


def configure(context: TraceContext) -> None:
    """Set the per-process trace settings (setup_tracing calls this)."""
    global _CONTEXT
    _CONTEXT = context


def kind_attributes(kind: str) -> dict[str, AttrValue]:
    """Span-kind attributes for both viewers (Phoenix and Langfuse)."""
    return {SPAN_KIND: kind, LANGFUSE_TYPE: _LANGFUSE_TYPES.get(kind, "span")}


def _allowed(attributes: Mapping[str, Any]) -> dict[str, Any]:
    if _CONTEXT.capture_content:
        return dict(attributes)
    return {k: v for k, v in attributes.items() if not is_content(k)}


# The conversation (thread) id of the request being handled. Copied onto EVERY
# span by the span processor: Langfuse only groups a session reliably when all
# spans carry it, not just the root. A ContextVar, so concurrent requests never
# see each other's id (asyncio.to_thread copies it into worker threads).
_SESSION: ContextVar[str | None] = ContextVar("docrag_session", default=None)


@contextmanager
def session(session_id: str) -> Iterator[None]:
    token = _SESSION.set(session_id)
    try:
        yield
    finally:
        _SESSION.reset(token)


def tracer() -> trace.Tracer:
    # Looked up per call (not cached at import), so it always uses the provider
    # installed by setup_tracing().
    return trace.get_tracer(TRACER_NAME)


def in_trace() -> bool:
    """True if there is an active, valid span (= we are inside a trace)."""
    return trace.get_current_span().get_span_context().is_valid


@contextmanager
def root_span(
    name: str, kind: str = CHAIN, attributes: Mapping[str, AttrValue] | None = None
) -> Iterator[Span]:
    """Start a new trace (or a child, if one is already active).

    Use this at the entry points of the query path; ``@timed`` spans below it
    then attach automatically via the context (contextvars).
    """
    # The span is current for everything inside the `with` (also across await and
    # asyncio.to_thread), ends on exit, and records an escaping exception as ERROR.
    with tracer().start_as_current_span(
        name, attributes={**kind_attributes(kind), **_allowed(attributes or {})}
    ) as span:
        yield span


@contextmanager
def child_of(
    parent: SpanContext | None,
    name: str,
    kind: str = CHAIN,
    attributes: Mapping[str, AttrValue] | None = None,
) -> Iterator[Span]:
    """Start a span under ``parent`` even if that span has already ended.

    The eval runner answers all questions first and judges them afterwards; this
    puts each judge call into its question's trace anyway.
    """
    ctx: Context | None = None
    if parent is not None and parent.is_valid:
        ctx = trace.set_span_in_context(trace.NonRecordingSpan(parent))
    with tracer().start_as_current_span(
        name,
        context=ctx,
        attributes={**kind_attributes(kind), **_allowed(attributes or {})},
    ) as span:
        yield span


def annotate(attributes: Mapping[str, Any], max_len: int = MAX_TEXT) -> None:
    """Add attributes to the current span. None values are skipped.

    Cheap no-op outside a trace, so domain code can call it unconditionally.
    Text attributes are dropped when content capture is off.
    """
    span = trace.get_current_span()
    if not span.is_recording():
        return
    for key, value in _allowed(attributes).items():
        if value is None:
            continue
        if isinstance(value, str) and len(value) > max_len:
            value = value[:max_len] + "...[truncated]"
        span.set_attribute(key, value)


def set_io(input: Any = None, output: Any = None) -> None:
    """Observation input/output as Langfuse and Phoenix show it (input.value /
    output.value). Non-strings are stored as JSON."""

    def text(value: Any) -> str | None:
        if value is None or isinstance(value, str):
            return value
        return json.dumps(value, ensure_ascii=False, default=str)

    annotate({"input.value": text(input), "output.value": text(output)})


def ranked(hits: Any, text_chars: int = 0) -> list[dict[str, Any]]:
    """A compact, readable view of ranked hits for an observation's output."""
    out = []
    for rank, h in enumerate(hits, start=1):
        row: dict[str, Any] = {"rank": rank, "id": h.chunk.id, "score": h.score}
        if text_chars:
            row["text"] = " ".join(h.chunk.text.split())[:text_chars]
        out.append(row)
    return out


def annotate_documents(
    prefix: str, docs: list[tuple[str, float, str]], max_content: int = 1000
) -> None:
    """OpenInference document list: ``<prefix>.<i>.document.{id,score,content}``.

    prefix is "retrieval.documents" for retrievers and
    "reranker.output_documents" for rerankers.
    """
    if not trace.get_current_span().is_recording():
        return
    attrs: dict[str, AttrValue] = {}
    for i, (doc_id, score, content) in enumerate(docs):
        attrs[f"{prefix}.{i}.document.id"] = doc_id
        attrs[f"{prefix}.{i}.document.score"] = float(score)
        attrs[f"{prefix}.{i}.document.content"] = content[:max_content]
    annotate(attrs)


def hit_docs(hits: Any) -> list[tuple[str, float, str]]:
    """(chunk id, score, text) for annotate_documents, from a list of Hits."""
    return [(h.chunk.id, h.score, h.chunk.text) for h in hits]


@dataclass(frozen=True)
class LangfuseTarget:
    """Where and how to send spans to Langfuse (OTLP/HTTP, Basic auth)."""

    host: str
    public_key: str
    secret_key: str

    @property
    def endpoint(self) -> str:
        return self.host.rstrip("/") + "/api/public/otel/v1/traces"

    @property
    def headers(self) -> dict[str, str]:
        pair = f"{self.public_key}:{self.secret_key}".encode()
        return {
            "Authorization": "Basic " + base64.b64encode(pair).decode("ascii"),
            # Without it Langfuse may show OTLP data up to 10 minutes late.
            "x-langfuse-ingestion-version": "4",
        }


def _sdk_disabled() -> bool:
    return os.environ.get("OTEL_SDK_DISABLED", "").strip().lower() == "true"


def setup_tracing(
    endpoint: str | None,
    service_name: str = "docrag",
    exporter: Any = None,
    langfuse: LangfuseTarget | None = None,
    context: TraceContext | None = None,
) -> bool:
    """Install a global TracerProvider that exports spans. Returns True if set up.

    ``endpoint``: OTLP/HTTP traces URL, e.g. http://127.0.0.1:6006/v1/traces
    (Phoenix). ``langfuse``: also export to Langfuse. ``exporter``: a ready
    SpanExporter (tests pass an InMemorySpanExporter). ``context``: environment,
    release, tags and content capture for every span. With no exporter, or
    with OTEL_SDK_DISABLED=true, nothing happens and tracing stays off.
    """
    if _sdk_disabled():
        log.info("tracing disabled by OTEL_SDK_DISABLED")
        return False
    if endpoint is None and exporter is None and langfuse is None:
        return False
    configure(context or TraceContext())
    # SDK imported only here: everything else depends on the small opentelemetry-api.
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import SpanLimits, TracerProvider
    from opentelemetry.sdk.trace.export import (
        BatchSpanProcessor,
        SimpleSpanProcessor,
    )

    provider = TracerProvider(
        resource=Resource.create({"service.name": service_name}),
        span_limits=SpanLimits(max_span_attributes=MAX_SPAN_ATTRIBUTES),
    )
    # Added first, so every exporter sees these attributes on every span.
    provider.add_span_processor(_context_processor())
    targets: list[str] = []
    if exporter is not None:
        # Simple = export synchronously when a span ends (tests can assert at once).
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        targets.append(type(exporter).__name__)
    if endpoint is not None or langfuse is not None:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )

        # Batch = buffer spans and export in a background thread, so a slow or
        # absent collector never adds latency to a request. The provider flushes
        # the buffer at interpreter exit (atexit), so CLI runs lose nothing.
        if endpoint is not None:
            provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint)))
            targets.append(endpoint)
        if langfuse is not None:
            otlp = OTLPSpanExporter(langfuse.endpoint, headers=langfuse.headers)
            provider.add_span_processor(BatchSpanProcessor(otlp))
            targets.append(langfuse.endpoint)  # the URL only, never the keys
    # The global provider can be set once per process; OTel warns on a 2nd call.
    trace.set_tracer_provider(provider)
    log.info("tracing enabled -> %s", ", ".join(targets))
    return True


def _context_processor() -> Any:
    """A SpanProcessor that stamps trace-level context on every new span.

    Langfuse filters and aggregates by session, environment, release and tags
    only if EVERY span carries them, and tags must be set when a span starts.
    Root spans also get a stable trace name (TRACE_NAMES).
    """
    from opentelemetry.sdk.trace import SpanProcessor

    class ContextSpanProcessor(SpanProcessor):
        def on_start(self, span: Any, parent_context: Context | None = None) -> None:
            ctx = _CONTEXT
            if ctx.environment:
                span.set_attribute("langfuse.environment", ctx.environment)
                span.set_attribute("deployment.environment.name", ctx.environment)
            if ctx.release:
                span.set_attribute("langfuse.release", ctx.release)
            if ctx.tags:
                span.set_attribute("langfuse.trace.tags", list(ctx.tags))
            session_id = _SESSION.get()
            if session_id is not None:
                span.set_attribute("session.id", session_id)
            parent = trace.get_current_span(parent_context).get_span_context()
            if not parent.is_valid:  # a root span: it names the trace
                name = TRACE_NAMES.get(span.name, span.name)
                span.set_attribute("langfuse.trace.name", name)

    return ContextSpanProcessor()
