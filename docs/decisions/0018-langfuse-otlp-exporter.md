# 0018. Langfuse as a second OTLP exporter, not the Langfuse SDK

- **Status:** Accepted
- **Date:** 2026-11-14 (`0810572`); audit fixes 2026-11-15 (`0a56310`, `88bfed9`)

## Context

Phoenix is good for local debugging. Langfuse adds what an LLM product team
uses: generations with tokens and cost, sessions, environments, a hosted UI.
The spans already exist ([0017](0017-opentelemetry-tracing.md)).

## Decision

Langfuse is a second OTLP/HTTP exporter on the same spans, enabled when
`LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` are set. Spans carry both
attribute sets: OpenInference for Phoenix and `gen_ai.*` plus `langfuse.*` for
Langfuse (observation type, role-labelled input, output, model, tokens, cost in
USD). A span processor stamps session, environment, release and the entry-point
tag on every span. `OTEL_SDK_DISABLED=true` turns all export off, and
`DOCRAG_TRACE_CONTENT=false` drops every text attribute.

## Alternatives considered

- **The Langfuse Python SDK (decorators):** a second instrumentation next to
  OTel, and code coupled to one vendor.
- **Replace Phoenix with Langfuse:** Phoenix runs locally without an account; both
  read the same spans, so there is no need to choose.

## Consequences

- **+** One instrumentation, two viewers; switching is environment variables only.
- **+** Audited against a live Langfuse Cloud project: the first real trace
  showed gaps in-memory tests could not (OTel's 128-attribute limit dropped the
  type keys on `search`; no generation input; FastAPI-internal spans). Fixed.
- **−** Langfuse features outside OTLP are not used: judge verdicts as scores,
  prompt management.
- **−** Langfuse Cloud is a third party that receives questions and document
  text unless content capture is off; hence off by default. See
  [0024](0024-local-langfuse-in-compose.md).

## Evidence

- `src/docrag/shared/tracing.py`, `src/docrag/llm/proxy.py`, `tests/unit/test_tracing.py`,
  `scripts/langfuse_trace_audit.py`, `docs/assets/langfuse-trace.png`
- [Lesson 14, section 6](../lessons/14_operability.md), [`docs/deploy.md`, Observability](../deploy.md)
