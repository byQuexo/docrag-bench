# 0006. Our own LLM proxy with a content-addressed cache and a budget cap

- **Status:** Accepted
- **Date:** 2026-11-01 (predates the git history; in `e1ca9c0`)

## Context

Answers, the judge and the vision-LLM OCR call several providers (Anthropic
directly, many models through OpenRouter). Benchmark runs make hundreds of
calls and are re-run often; spend must be bounded and every number must be
reproducible.

## Decision

`LLMProxy` sits in front of a `Provider` ABC with two adapters: an
OpenAI-compatible one on httpx (OpenRouter, local servers) and the official
Anthropic SDK. It enforces one message order, asks for structured output
(Pydantic schema, one retry with the validation error), caches completions in
SQLite under a hash of route, messages, schema and temperature, records billed
cost, and stops at `DOCRAG_BUDGET_EUR`.

## Alternatives considered

- **LiteLLM:** the pragmatic choice in a product with many providers. Here the
  gateway logic (format mapping, ordering, caching, cost, tracing) is what the
  project demonstrates, and with 2–3 adapters it is small and fully tested.
  LiteLLM could replace the adapters behind the same `Provider` interface.
- **The openai SDK for the compatible adapter:** the conversion is the point of
  the module, and `httpx.MockTransport` makes it easy to test.
- **Regex JSON repair:** guesses intent silently; failures are counted instead.
- **Redis as the cache:** a single-machine benchmark needs no server; SQLite is
  in the standard library and persists across runs.

## Consequences

- **+** A rerun of the same config hits the cache: same numbers, 0 EUR. A
  changed prompt or retrieval result changes the key, so stale answers cannot
  be replayed; this is what makes the CI cache safe ([0010](0010-ci-eval-gate.md)).
- **+** One place for cost, budget and tracing ([0018](0018-langfuse-otlp-exporter.md)).
- **−** We maintain provider quirks ourselves (system prompt placement,
  `output_config`, refusals, `require_parameters` on OpenRouter).
- **−** Provider-side model drift is invisible on cache hits.

## Evidence

- `src/docrag/llm/proxy.py`, `src/docrag/llm/openai_compat.py`,
  `src/docrag/llm/anthropic_provider.py`, `src/docrag/shared/cache.py`
- [Lesson 05](../lessons/05_llm_proxy.md), [interview prep 10–11, 15, D2.4](../lessons/10_interview_prep.md)
