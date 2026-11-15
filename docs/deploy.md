# Deploying docrag

Three ways to run the API, from simplest to closest to production. All three use
the same image (`Dockerfile`) and the same settings (environment variables, see
[`.env.example`](../.env.example) and `src/docrag/shared/settings.py`).

| | What runs | Good for |
|---|---|---|
| `docker compose` | Qdrant + API + Phoenix (traces) | working on the app locally |
| kind | Qdrant StatefulSet + API Deployment in a local Kubernetes cluster | testing the manifests, probes and security settings |
| a real cluster | the same manifests (`deploy/k8s/base`) | a deployment, after the gaps under "Before production" are closed |

## The image

Multi-stage build: Node builds the web UI, a uv stage builds the virtualenv, and
the final `python:3.12-slim` stage gets only the UI files and the `.venv`. It
runs as the unprivileged user `app` (uid 10001), writes only to `/app/models`
and `/app/.cache`, and logs JSON (`DOCRAG_LOG_FORMAT=json`). CI builds it on
every push and writes its size to the job summary (job `deploy-smoke`).

It is not small. Two numbers, because they differ: CI's `docker image inspect`
reported 365 MB (probably the compressed size; the CI step now prints both),
and a local build measured about 940 MB of files inside the image (2026-11-14),
nearly all of it the Python environment: polars (170 MB), OpenCV + RapidOCR (~185 MB, only
needed for OCR at ingest time), PyMuPDF (116 MB), ONNX Runtime (68 MB).
scikit-learn and SciPy (~190 MB) were removed from the runtime dependencies for
that reason. Next step: an `ocr` extra so the serving image skips OpenCV.

```bash
docker build -t docrag-bench:local .
```

ONNX models (~560 MB) are not baked into the image; the API downloads them into
`/app/models` on first start. That keeps the image small and rebuilds fast, at
the cost of a slow first start (the start-up probes allow for it).

## 1. docker compose

```bash
docker compose up -d qdrant          # only the vector DB, e.g. for CLI ingest/eval
docker compose up -d                 # + API on http://localhost:8000 + Phoenix on :6006
docker compose ps                    # qdrant shows (healthy); the API waits for it
```

The API needs an index and an LLM key to answer: ingest first
(`uv run docrag ingest --config configs/baseline.yaml`) and put
`OPENROUTER_API_KEY` into `.env`. All ports are bound to 127.0.0.1 only.

## 2. kind (local Kubernetes)

Prerequisites: Docker, [kind](https://kind.sigs.k8s.io/) (`winget install Kubernetes.kind`) and kubectl.

```bash
kind create cluster --name docrag    # cgroup v1 host? add: --config deploy/k8s/kind-cluster.yaml
docker build -t docrag-bench:kind .
kind load docker-image docrag-bench:kind --name docrag
kubectl apply -k deploy/k8s/overlays/kind
bash deploy/k8s/smoke.sh             # rollout, /healthz, /readyz, one POST /query
```

If `kind create cluster` fails with `context deadline exceeded`, check
`docker info --format '{{.CgroupVersion}}'`: Kubernetes 1.35+ refuses to run the
kubelet on cgroup v1, which older WSL 2 kernels under Docker Desktop still use.
`deploy/k8s/kind-cluster.yaml` allows it for a local cluster; switching WSL to
cgroup v2 is the cleaner fix (see the comment in that file).

The kind overlay serves `configs/e2e_fake.yaml` (fake embedder, reranker and
LLM) on an empty index, so it needs no model download and no API key, and the
smoke test expects an abstention. It tests the deployment, not answer quality;
that is what the eval gate in CI is for. CI runs exactly these steps on every
push (job `deploy-smoke`).

Look around and clean up:

```bash
kubectl -n docrag get pods,svc,pvc
kubectl -n docrag logs deploy/docrag-api      # JSON lines with trace_id
kubectl -n docrag port-forward svc/docrag-api 8080:80
kind delete cluster --name docrag
```

## 3. A real cluster

`deploy/k8s/base` is the production-shaped variant: it serves
`configs/answer_haiku_vlm_v3.yaml` and expects a Secret for the LLM key.

```bash
kubectl create namespace docrag
kubectl -n docrag create secret generic docrag-secrets \
  --from-literal=OPENROUTER_API_KEY=<key>      # optional: LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY
kubectl apply -k deploy/k8s/base               # set the image under `images:` first
```

### What the manifests do, and why

| Setting | Why |
|---|---|
| init container waits for Qdrant's `/readyz` | the API creates its collection at start-up; without the wait the first rollout crash-looped twice until Qdrant was up |
| `startupProbe` on `/healthz`, up to 10 min | the server only listens after the models are loaded; a slow download must not count as a hang |
| `livenessProbe` on `/healthz` | restarts a hung process. Checks no dependency: if Qdrant is down, restarting the API fixes nothing and would restart every replica in a loop |
| `readinessProbe` on `/readyz` | Qdrant reachable and the collection exists; otherwise the pod gets no traffic |
| `replicas: 1` | conversation threads live in process memory; a second replica would not know them |
| `maxUnavailable: 0`, `preStop: sleep 5`, 30 s grace | rolling updates without dropped requests: the old pod serves until the new one is ready, leaves the Service before SIGTERM, and uvicorn finishes in-flight requests (20 s) |
| `readOnlyRootFilesystem`, non-root, all capabilities dropped, seccomp `RuntimeDefault` | the namespace enforces the `restricted` Pod Security Standard |
| requests/limits | a starting point, not a measurement under load: check `kubectl top pod` and adjust |
| Qdrant as StatefulSet with a 5 Gi volume claim | the index survives pod restarts; the `-unprivileged` image runs as uid 1000 |

### Before production (not done here)

- **Ingest** has no Job or CronJob: the index must be filled with `docrag ingest`
  from a machine that can reach Qdrant.
- **No ingress, TLS or auth** in front of the API, and no Qdrant API key.
- **Threads in memory**: one replica only, and conversations are lost on restart.
- **Models in an `emptyDir`**: re-downloaded whenever the pod is replaced. A
  PVC or an image with the models baked in avoids that.
- No NetworkPolicy, no HorizontalPodAutoscaler, no backups of the Qdrant volume.

## Observability

- **Logs**: JSON lines on stdout. Each has `trace_id`/`span_id` when it was
  written inside a request, so a log line leads to its trace. One access line
  per request (`method`, `path`, `status`, `duration_ms`); probes are not logged.
- **Traces**: OpenTelemetry, off unless an exporter is configured:
  `DOCRAG_OTEL_ENDPOINT` (Phoenix, any OTLP/HTTP collector) and/or
  `LANGFUSE_PUBLIC_KEY` + `LANGFUSE_SECRET_KEY` (+ `LANGFUSE_HOST`).
  `OTEL_SDK_DISABLED=true` turns all of them off. A query becomes one trace,
  named `answer-question`: HTTP request (question in, answer out) → `query` →
  `answer` → `retrieve` (`embed`, `search`, `rerank`) → `llm` (role-labelled
  messages, model, tokens, cost) → `verify`. Every span carries the session
  (thread id), `DOCRAG_ENVIRONMENT` (`production` in `deploy/k8s/base`),
  `DOCRAG_RELEASE` and the entry point as a tag (`api`, `ask`, `eval`).
  `DOCRAG_TRACE_CONTENT=false` keeps questions, prompts, answers and document
  text out of the traces (structure, timings, ids, scores, tokens and cost
  stay). Checked against a live Langfuse Cloud project on 2026-11-15;
  `scripts/langfuse_trace_audit.py` prints the latest trace as a tree for the
  same check.
- **Timeouts and retries**: see the `DOCRAG_*_TIMEOUT_S` / `*_MAX_RETRIES`
  settings. `/query` answers 504 past `DOCRAG_REQUEST_TIMEOUT_S` and 503 when
  Qdrant is unreachable after its retries.
