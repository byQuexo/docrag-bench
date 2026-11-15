# 0022. A shared thread store behind a flag

- **Status:** Proposed
- **Date:** 2026-11-15

## Problem

Conversation threads live in `InMemoryThreadStore` (a dict with a lock, last 20
messages per thread). That pins the API to `replicas: 1`
([0021](0021-kubernetes-kustomize-kind.md)): a second replica would answer a
follow-up with 404, and every restart or rollout loses all conversations. The
`ThreadStore` Protocol (`create`, `get`, `append`) already separates the API
from the storage.

## Options

| Option | For | Against |
|---|---|---|
| **Redis** | Fast; TTL per thread matches "a conversation expires"; one small container; the cache could move there later too (lesson 01) | New infrastructure; durability depends on persistence settings; not queryable for analysis |
| **Postgres** | Durable, queryable (threads become data), the usual default in a product | Heavier to run; needs a schema and migrations; TTL is a cleanup job |
| **SQLite on a volume** | No new service | Still one writer; does not allow a second replica, so it solves restarts but not scaling |
| **Sticky sessions** | No code change | Fragile: a pod restart still loses threads; ingress-specific |

Common to Redis and Postgres: a setting such as
`DOCRAG_THREAD_STORE=memory|redis|postgres` (default `memory`, so tests and the
benchmark stay as they are), a connection URL in `Settings`, an integration test
with testcontainers ([0019](0019-testcontainers-for-qdrant.md)), a readiness
check that includes the store ([0015](0015-liveness-vs-readiness.md)), and
retries under the existing policy ([0014](0014-retries-and-deadlines.md)).

## Open questions for the decision

- Must conversations survive days (durable) or only a session (TTL)?
- Will threads ever be analysed, exported or deleted per user (GDPR)?
- Should the LLM cache move to the same store for multi-replica cost savings?
- After this: raise `replicas` and add a PodDisruptionBudget?

## Evidence

- `src/docrag/threads/store.py`, `src/docrag/api/app.py`, `deploy/k8s/base/api.yaml`
- [Lesson 07, section 5](../lessons/07_api_and_threads.md), [interview prep 16](../lessons/10_interview_prep.md), [`docs/deploy.md`, Before production](../deploy.md)
