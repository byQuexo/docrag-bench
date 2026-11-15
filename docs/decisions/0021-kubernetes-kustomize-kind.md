# 0021. Kubernetes manifests with kustomize, tested on kind in CI, one replica

- **Status:** Accepted
- **Date:** 2026-11-14 (`7fe38a9`); in-memory thread store since 2026-11-01 (`e1ca9c0`)

## Context

A Dockerfile alone does not show that the service runs under real orchestration:
probes, rolling updates, security policy, a stateful dependency. Conversation
threads live in process memory (`InMemoryThreadStore` behind a `ThreadStore`
Protocol).

## Decision

- **Manifests:** `deploy/k8s/base` (API Deployment, Qdrant StatefulSet with a
  5 Gi volume, ConfigMap, probes, resources, init container waiting for Qdrant,
  `restricted` Pod Security Standard) and `overlays/kind` (fake models, no key),
  composed with kustomize.
- **CI:** job `deploy-smoke` builds the image, loads it into a kind cluster,
  applies the overlay and runs `deploy/k8s/smoke.sh` (rollout, both probes, one
  `/query`, security context).
- **`replicas: 1`**, `maxUnavailable: 0`, `preStop` sleep and uvicorn graceful
  shutdown for rolling updates without dropped requests.

## Alternatives considered

- **More replicas:** a follow-up question could land on a pod that never saw the
  thread and get a 404. Scaling needs a shared store first ([0022](0022-shared-thread-store.md)).
- **Compose only:** tests neither probes nor the security policy.
- **Helm, minikube or k3d:** not compared in writing. **Begründung offen:** why
  kustomize rather than Helm, and kind rather than minikube or k3d.

## Consequences

- **+** The real manifests are deployed on every push; failures print events,
  pod descriptions and logs.
- **+** Found real issues: the API crash-looped until Qdrant was ready (fixed by
  the init container); Kubernetes 1.35+ refuses cgroup v1 hosts (local
  `kind-cluster.yaml`); Docker Hub rate limits in CI (images loaded into kind).
- **−** One replica: no horizontal scaling, and threads are lost on restart.
- **−** Not covered: ingress, TLS, auth, NetworkPolicy, autoscaling, Qdrant
  backups, ingest as a Job.

## Evidence

- `deploy/k8s/`, `.github/workflows/ci.yml` (job `deploy-smoke`), `src/docrag/threads/store.py`
- [`docs/deploy.md`](../deploy.md), [lesson 07](../lessons/07_api_and_threads.md), [lesson 14, section 9](../lessons/14_operability.md)
