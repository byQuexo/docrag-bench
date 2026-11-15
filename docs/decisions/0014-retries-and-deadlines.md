# 0014. Retries with full jitter and Retry-After; one deadline per request (504)

- **Status:** Accepted
- **Date:** 2026-11-14 (`b5d68d8` retries and timeouts, `6fe9283` deadline, `0c78cdc` 503)

## Context

LLM providers return 429s and 5xx, and connections drop. Qdrant can be briefly
unreachable. Before this, a failure surfaced as a 500, and nothing bounded how
long a `/query` could take.

## Decision

- **One policy** in `shared/retry.py` for LLM and Qdrant calls: retry only
  transient errors (connection errors, timeouts, 429, 5xx), and only
  idempotent calls (searches, counts, upserts with deterministic point ids).
- **Backoff:** exponential with full jitter, `random(0, base · 2^attempt)`;
  `Retry-After` wins, capped at 20 s.
- **Configurable** via `Settings`: `DOCRAG_LLM_TIMEOUT_S` (120),
  `DOCRAG_LLM_MAX_RETRIES` (2), `DOCRAG_QDRANT_TIMEOUT_S` (30),
  `DOCRAG_QDRANT_MAX_RETRIES` (2). The Anthropic SDK gets the same values for
  its built-in retries.
- **Deadline:** `/query` runs under `asyncio.timeout(DOCRAG_REQUEST_TIMEOUT_S)`
  (150 s) and answers **504** when it runs out; Qdrant still failing after its
  retries is a **503**.

## Alternatives considered

- **tenacity:** the standard library for this and would be fine. The policy is
  ~40 lines, both call sites need the same three decisions, and tests pin it
  exactly with injected `sleep` and `random`.
- **Retry every error:** a 400 or a missing collection (404) is a bug or a
  config error; retrying only delays the real message.
- **Plain exponential backoff:** clients that failed together retry together.
- **No overall deadline:** retries multiply latency without bound.

## Consequences

- **+** Transient faults are absorbed; config errors fail fast (a missing
  collection fails in under a second, tested).
- **+** Clients and monitoring can tell 4xx, 502 (provider), 503 (search
  backend), 504 (deadline) apart.
- **−** Worst-case latency grows with retries, bounded only by the deadline.
- **−** Our own retry code to maintain instead of a library.

## Evidence

- `src/docrag/shared/retry.py`, `src/docrag/llm/openai_compat.py`, `src/docrag/store/qdrant.py`,
  `src/docrag/api/app.py`, `tests/unit/test_retry.py`, `tests/unit/test_api_ops.py`
- [Lesson 14, section 4](../lessons/14_operability.md)
