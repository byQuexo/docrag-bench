# 0024. Langfuse locally in `compose.yaml`

- **Status:** Proposed
- **Date:** 2026-11-15

## Problem

Langfuse is only reachable as Langfuse Cloud ([0018](0018-langfuse-otlp-exporter.md)):
it needs an account and keys, and it sends questions, prompts and document text
to a third party unless `DOCRAG_TRACE_CONTENT=false`. Phoenix runs locally in
`compose.yaml`, so a contributor without an account sees only Phoenix.

## Options

| Option | For | Against |
|---|---|---|
| **Langfuse services in `compose.yaml`, behind a compose profile** (`docker compose --profile langfuse up`) | Traces stay on the machine; the same exporter code, only `LANGFUSE_HOST` changes; no account needed | Langfuse v3 self-hosting is several services (web, worker, Postgres, ClickHouse, Redis, S3-compatible storage, per the Langfuse self-hosting docs; to be confirmed against the current version); much heavier than Phoenix; keys still have to be created in the local UI |
| **Keep Cloud only** | Nothing to run | Third party by default; needs an account |
| **Phoenix only for local, Langfuse only in deployments** | Light local setup | The Langfuse view (generations, cost, sessions) is not available locally |

Whatever is chosen: the profile must be opt-in, so `docker compose up -d`
stays light, and the ports bind to 127.0.0.1 like the rest of `compose.yaml`.

## Open questions for the decision

- Is the extra memory and start-up time acceptable on the development laptop?
- Should the local keys be created by hand in the UI or provisioned at start-up?
- Does the README then claim "verified against local and Cloud Langfuse", and
  which one does CI use (currently neither)?

## Evidence

- `compose.yaml` (Phoenix), `src/docrag/shared/tracing.py` (`LangfuseTarget`),
  `.env.example` (`LANGFUSE_HOST`)
- [Lesson 14, section 6](../lessons/14_operability.md), [`docs/deploy.md`, Observability](../deploy.md)
