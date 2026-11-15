# 0015. `/healthz` checks only the process, `/readyz` checks Qdrant

- **Status:** Accepted
- **Date:** 2026-11-14 (`6fe9283`; Kubernetes probes in `7fe38a9`)

## Context

Docker and Kubernetes need to know two different things: is the process hung
(restart it), and can it serve right now (send it traffic). The API had one
`/health` endpoint. Its main dependency, Qdrant, can be down while the API
process is fine.

## Decision

- **`/healthz` (liveness):** 200 if the process serves HTTP. Checks nothing
  else. Used by the Docker `HEALTHCHECK` and the Kubernetes liveness and
  startup probes.
- **`/readyz` (readiness):** 200 if Qdrant answers within 2 s
  (`DOCRAG_READINESS_TIMEOUT_S`) and the collection exists, else 503 with the reason.
- Probes are excluded from traces and the access log.

## Alternatives considered

- **One endpoint that checks Qdrant, used for liveness too:** if Qdrant is down,
  every replica fails liveness and is restarted in a loop, and restarting fixes
  nothing.
- **Readiness without a timeout:** a hanging Qdrant would hang the probe.

## Consequences

- **+** A Qdrant outage takes the pod out of the Service without restarting it,
  and it returns once Qdrant is back (checked on kind).
- **+** Probe traffic does not drown the logs and traces.
- **−** A process that is alive but broken in another way (e.g. a stuck model)
  is not detected by liveness.

## Evidence

- `src/docrag/api/app.py`, `tests/unit/test_api_ops.py`, `Dockerfile` (`HEALTHCHECK`),
  `deploy/k8s/base/api.yaml` (probes)
- [Lesson 14, section 3](../lessons/14_operability.md), [`docs/deploy.md`](../deploy.md)
