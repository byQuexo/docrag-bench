"""OpenTelemetry tracing for the question -> answer path, viewed in Phoenix.

Design (see docs/lessons/13_tracing.md):

* Every ``@timed`` function also opens a span (shared/timing.py). Call sites
  didn't change when tracing was added.
* A span is only recorded INSIDE a trace. Traces start at explicit roots: an API
  request, one eval question, ``docrag ask``. Ingest (parse, embed, upsert) has no
  root, so it produces no spans, even when it runs inside ``docrag eval``.
* Exporting is opt-in: ``setup_tracing()`` is only called by serve/ask/eval and
  only when ``DOCRAG_OTEL_ENDPOINT`` is set. Without it the OpenTelemetry API is
  a no-op (one function call, no allocation worth measuring).

Attribute names follow OpenInference (https://github.com/Arize-ai/openinference),
the semantic conventions Phoenix understands: it shows LLM spans with tokens and
the prompt, retriever spans with a document list, and so on.
"""

import logging
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
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

# Long prompts/contexts are truncated: a span is not a log archive, and Phoenix
# gets slow with megabyte-sized attributes.
MAX_TEXT = 8_000

AttrValue = str | bool | int | float


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
        name, attributes={SPAN_KIND: kind, **(attributes or {})}
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
        name, context=ctx, attributes={SPAN_KIND: kind, **(attributes or {})}
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


def setup_tracing(
    endpoint: str | None,
    service_name: str = "docrag",
    exporter: Any = None,
) -> bool:
    """Install a global TracerProvider that exports spans. Returns True if set up.

    ``endpoint``: OTLP/HTTP traces URL, e.g. http://127.0.0.1:6006/v1/traces
    (Phoenix). ``exporter``: a ready SpanExporter instead (tests pass an
    InMemorySpanExporter). With neither, nothing happens and tracing stays off.
    """
    if endpoint is None and exporter is None:
        return False
    # SDK imported only here: everything else depends on the small opentelemetry-api.
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import (
        BatchSpanProcessor,
        SimpleSpanProcessor,
    )

    provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
    if exporter is not None:
        # Simple = export synchronously when a span ends (tests can assert at once).
        provider.add_span_processor(SimpleSpanProcessor(exporter))
    else:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )

        # Batch = buffer spans and export in a background thread, so a slow or
        # absent collector never adds latency to a request. The provider flushes
        # the buffer at interpreter exit (atexit), so CLI runs lose nothing.
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint)))
    # The global provider can be set once per process; OTel warns on a 2nd call.
    trace.set_tracer_provider(provider)
    log.info("tracing enabled -> %s", endpoint or type(exporter).__name__)
    return True
