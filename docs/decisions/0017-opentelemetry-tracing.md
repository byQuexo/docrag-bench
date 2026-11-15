# 0017. OpenTelemetry as the tracing layer, query path only

- **Status:** Accepted
- **Date:** 2026-11-01 (milestone M6, in `e1ca9c0`)

## Context

`@timed` already gave p50/p95 per step for `run.json`, but not the story of a
single answer: which chunks were retrieved, what the prompt was, which
citation failed and why. The project should not be tied to one tracing vendor.

## Decision

We instrument with the OpenTelemetry API: `@timed` also opens a span when it
runs inside a trace, so request → `answer` → `retrieve` (`embed`, `search`,
`rerank`) → `llm` → `verify` becomes one trace. Spans carry OpenInference
attributes so Phoenix shows span kinds and LLM messages. Exporting is off
unless `DOCRAG_OTEL_ENDPOINT` is set; Phoenix runs in `compose.yaml`. The API,
`eval` and `ask` are traced; ingest is not.

## Alternatives considered

- **A vendor SDK (LangSmith, Langfuse):** couples the code to one backend.
  With OTel the code depends on the API only and the backend is a config value
  (Phoenix, Jaeger, Tempo, Datadog, Langfuse).
- **Tracing ingest too:** an offline batch job; a span per page or chunk would
  flood the store with spans nobody reads. `@timed` and logs cover it.

## Consequences

- **+** FastAPI emits OTel natively, so HTTP and RAG spans land in one trace.
- **+** Adding Langfuse later was an exporter, not a rewrite ([0018](0018-langfuse-otlp-exporter.md)).
- **−** No sampling, no OTel Collector and no metrics yet; traces are for single
  requests, trends would need metrics.
- **−** Span names double as latency keys in `run.json`, so renaming one changes both.

## Evidence

- `src/docrag/shared/tracing.py`, `src/docrag/shared/timing.py`, `compose.yaml` (Phoenix)
- [Lesson 13](../lessons/13_tracing.md)
