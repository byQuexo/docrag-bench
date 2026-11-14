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

AttrValue = str | bool | int | float


def kind_attributes(kind: str) -> dict[str, AttrValue]:
    """Span-kind attributes for both viewers (Phoenix and Langfuse)."""
    return {SPAN_KIND: kind, LANGFUSE_TYPE: _LANGFUSE_TYPES.get(kind, "span")}


# The conversation (thread) id of the request being handled. Copied onto EVERY
# span by _SessionSpanProcessor: Langfuse only groups a session reliably when
# all spans carry it, not just the root. A ContextVar, so concurrent requests
# never see each other's id (asyncio.to_thread copies it into worker threads).
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
        name, attributes={**kind_attributes(kind), **(attributes or {})}
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
        name, context=ctx, attributes={**kind_attributes(kind), **(attributes or {})}
    ) as span:
        yield span


def annotate(attributes: Mapping[str, Any]) -> None:
    """Add attributes to the current span. None values are skipped.

    Cheap no-op outside a trace, so domain code can call it unconditionally.
    """
    span = trace.get_current_span()
    if not span.is_recording():
        return
    for key, value in attributes.items():
        if value is None:
            continue
        if isinstance(value, str) and len(value) > MAX_TEXT:
            value = value[:MAX_TEXT] + "...[truncated]"
        span.set_attribute(key, value)


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
) -> bool:
    """Install a global TracerProvider that exports spans. Returns True if set up.

    ``endpoint``: OTLP/HTTP traces URL, e.g. http://127.0.0.1:6006/v1/traces
    (Phoenix). ``langfuse``: also export to Langfuse. ``exporter``: a ready
    SpanExporter (tests pass an InMemorySpanExporter). With none of them, or
    with OTEL_SDK_DISABLED=true, nothing happens and tracing stays off.
    """
    if _sdk_disabled():
        log.info("tracing disabled by OTEL_SDK_DISABLED")
        return False
    if endpoint is None and exporter is None and langfuse is None:
        return False
    # SDK imported only here: everything else depends on the small opentelemetry-api.
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import (
        BatchSpanProcessor,
        SimpleSpanProcessor,
    )

    provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
    # Added first, so every exporter sees the session id on every span.
    provider.add_span_processor(_session_processor())
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


def _session_processor() -> Any:
    """A SpanProcessor that stamps the current session id on every new span."""
    from opentelemetry.sdk.trace import SpanProcessor

    class SessionSpanProcessor(SpanProcessor):
        def on_start(self, span: Any, parent_context: Context | None = None) -> None:
            session_id = _SESSION.get()
            if session_id is not None:
                span.set_attribute("session.id", session_id)

    return SessionSpanProcessor()
