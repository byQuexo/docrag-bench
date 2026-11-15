# 0016. JSON logs from a stdlib formatter, with the trace id, not structlog

- **Status:** Accepted
- **Date:** 2026-11-14 (`6fe9283`)

## Context

In a container, logs go to stdout and are read by machines. Text logs are hard
to filter, and a log line without a link to its trace cannot be followed. All
code already logs through `logging.getLogger(__name__)`.

## Decision

`shared/logs.py` has a ~40-line `JsonFormatter`: one JSON object per line with
`ts`, `level`, `logger`, `msg`, extras, the exception, and the current
OpenTelemetry `trace_id`/`span_id`. `DOCRAG_LOG_FORMAT=json` (set in the
image), `text` locally. uvicorn runs with `log_config=None` so it uses the same
handler, and one access line per request is written by our middleware.

## Alternatives considered

- **structlog / python-json-logger:** would add a dependency and, for structlog,
  a second logging API next to the stdlib calls already in every module.
- **uvicorn's own access log:** its own format, and it logs probes.

## Consequences

- **+** From a log line straight to its trace (Phoenix or Langfuse).
- **+** No new dependency; every existing `log.info(...)` is structured for free.
- **−** No processor pipeline as in structlog (e.g. redaction or context binding
  would be our own code).
- **−** Library quirks are ours to handle: uvicorn's `color_message` extra put
  ANSI codes into the JSON lines until it was excluded by hand.

## Evidence

- `src/docrag/shared/logs.py`, `src/docrag/cli.py`, `tests/unit/test_logs.py`
- [Lesson 14, section 5](../lessons/14_operability.md)
