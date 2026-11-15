# Lesson 14: Making the service operable

**Goal:** understand what was added to turn the API into something you can run,
watch and roll back, and be able to defend each choice as "X, not Y, because...".
No retrieval or answer logic changed in this step.

Files: `pyproject.toml`, `.pre-commit-config.yaml`, `src/docrag/shared/settings.py`,
`src/docrag/shared/retry.py`, `src/docrag/shared/logs.py`, `src/docrag/shared/tracing.py`,
`src/docrag/api/app.py`, `src/docrag/store/qdrant.py`, `src/docrag/llm/openai_compat.py`,
`.github/workflows/ci.yml`, `configs/ci_gate*.yaml`, `fixtures/ci_gate.yaml`,
`scripts/eval_gate.py`, `scripts/download_gate_data.py`, `Dockerfile`, `compose.yaml`,
`deploy/k8s/`, `docs/deploy.md`, `tests/e2e/test_e2e.py`, `tests/unit/test_api_ops.py`,
`tests/unit/test_retry.py`, `tests/unit/test_logs.py`, `tests/unit/test_eval_gate.py`.

There is no notebook for this lesson: most of it is configuration you read and
run, not functions you write.

This lesson tells the story. The same decisions are also recorded as short
lookup entries in [`docs/decisions/`](../decisions/README.md) (ADRs: context,
alternatives, consequences, evidence): section 1 → [0012](../decisions/0012-tooling-uv-mypy-precommit.md),
3 → [0015](../decisions/0015-liveness-vs-readiness.md),
4 → [0013](../decisions/0013-sync-qdrant-in-worker-threads.md) and [0014](../decisions/0014-retries-and-deadlines.md),
5 → [0016](../decisions/0016-stdlib-json-logs.md),
6 → [0018](../decisions/0018-langfuse-otlp-exporter.md),
7 → [0010](../decisions/0010-ci-eval-gate.md) and [0011](../decisions/0011-real-judge-in-ci.md),
8 → [0019](../decisions/0019-testcontainers-for-qdrant.md),
9 → [0020](../decisions/0020-container-image.md) and [0021](../decisions/0021-kubernetes-kustomize-kind.md).
The next steps are proposed in [0022](../decisions/0022-shared-thread-store.md)–[0024](../decisions/0024-local-langfuse-in-compose.md).

---

## 1. Quality tooling: one source of versions

- **mypy `--strict` on `src`, `tests` and `scripts`**, plus the pydantic plugin with
  `init_typed = true`. Without it, `Model(field=...)` accepts anything (the
  plugin types the constructor as `**Any`). Turning it on found a real mismatch:
  the Qdrant store built `Hit(source=...)` from a plain `str` where the model
  wants `Literal["dense", "bm25", "hybrid", "rerank"]`.
- **ruff** gained `ASYNC` (blocking calls inside `async def`), `RUF`, `SIM` and
  `PT` (e.g. `pytest.raises(ValueError)` must say *which* ValueError via `match=`).
  Two rules are switched off with a written reason in `pyproject.toml`, not with
  `noqa` comments scattered over the code.
- **pre-commit** runs ruff, mypy and `uv lock --check` as *local* hooks through
  `uv run`. *Why not the usual `ruff-pre-commit` repo hook?* It pins its own ruff
  version, which drifts from `uv.lock`; then a commit passes locally and fails in
  CI. With `uv run` there is exactly one version: the locked one.

## 2. Configuration: one Settings class, two kinds of config

Unchanged principle (lesson 01): `Settings` = environment (where, secrets,
timeouts), `ExperimentConfig` = what is measured. New fields are all *operational*:
`DOCRAG_LOG_FORMAT`, `*_TIMEOUT_S`, `*_MAX_RETRIES`, `LANGFUSE_*`. pydantic
validates them at start-up (`DOCRAG_LLM_TIMEOUT_S=-1` fails before the server
listens, thanks to `Field(gt=0)`).

## 3. Probes: liveness is not readiness

```
/healthz  -> 200 if the process serves HTTP. Checks nothing else.
/readyz   -> 200 if Qdrant answers within 2 s AND the collection exists, else 503.
```

*Why must liveness not check Qdrant?* Kubernetes **restarts** a pod whose liveness
probe fails. If Qdrant is down, every API replica would fail liveness and be
restarted, again and again, and the restart fixes nothing. Readiness only takes
the pod **out of the Service** until Qdrant is back. That is the right reaction to
a dependency outage.

`/readyz` runs the (sync) Qdrant call in a thread under `asyncio.wait_for`, so a
hanging Qdrant cannot hang the probe. Probes are excluded from traces and from
the access log, or they would be most of both.

## 4. Timeouts, retries, deadlines

`shared/retry.py` holds one policy for LLM and Qdrant calls:

- **retry only transient errors**: connection errors, timeouts, 429 and 5xx. A 400
  or "collection not found" (404) is a bug or a config error; retrying only delays
  the real message. (`tests/e2e`: a missing collection fails in < 1 s.)
- **exponential backoff with full jitter**: wait `random(0, base * 2**attempt)`.
  *Why jitter?* When an upstream hiccups, all in-flight requests fail at the same
  moment; without jitter they all retry at the same moment again.
- **`Retry-After` wins** (a 429 usually says when the window resets), capped at 20 s.
- **only idempotent calls are retried**: searches, counts, and upserts with
  deterministic point IDs. Retrying a non-idempotent call can do it twice.
- **one deadline per request** (`asyncio.timeout(DOCRAG_REQUEST_TIMEOUT_S)`):
  retries multiply latency, so the whole `/query` gets a budget and answers **504**
  when it runs out. A Qdrant failure after its retries is a **503**, not a 500.

*Why not tenacity?* It is the standard library for this and would be fine. Here
the whole policy is ~40 lines, both call sites need the same three decisions, and
the tests pin it down exactly (injected `sleep` and `random`).

*Why is Qdrant still a sync client in a worker thread, not `AsyncQdrantClient`?*
Retrieval is embed (CPU, ONNX) → search (I/O) → rerank (CPU) in one function.
The CPU parts must leave the event loop anyway (`asyncio.to_thread`), so the
search rides along in the same thread. An async client would split that function
in three for no measurable gain. The LLM calls, which wait seconds, are async.

## 5. Logs: JSON with the trace id

`DOCRAG_LOG_FORMAT=json` (set in the Docker image) writes one JSON object per
line. Every line written inside a request carries `trace_id`/`span_id`, so a log
line leads straight to its trace in Phoenix or Langfuse. One access line per
request: `method`, `path`, `status`, `duration_ms`. uvicorn is started with
`log_config=None` so it uses the same handler instead of its own format.

*Why not structlog?* All code already logs via `logging.getLogger(__name__)`. A
40-line `Formatter` keeps that and adds no dependency.

## 6. Tracing: one instrumentation, two viewers

Spans existed already (lesson 13). New:

- **Langfuse** is a second OTLP exporter on the same spans, not a second SDK.
  Langfuse reads `langfuse.observation.type` (`generation` for LLM calls) and the
  OpenTelemetry `gen_ai.*` conventions (model, tokens, cost in USD); Phoenix reads
  OpenInference (`openinference.span.kind`, `llm.*`). Both sets are written.
- **Session on every span**: Langfuse groups a conversation reliably only when
  every span carries `session.id`. A tiny `SpanProcessor` stamps it from a
  `ContextVar` that the `/query` handler sets; `asyncio.to_thread` copies the
  ContextVar into the worker thread, so `embed`/`search`/`rerank` get it too.
- **Switches**: nothing is exported unless `DOCRAG_OTEL_ENDPOINT` or both Langfuse
  keys are set; `OTEL_SDK_DISABLED=true` turns everything off.

*Privacy*: spans contain the question, the prompt and document text. Langfuse
Cloud is a third party, which is why it is off by default, and
`DOCRAG_TRACE_CONTENT=false` drops every text attribute while keeping the
structure, ids, scores, tokens and cost.

**Audited against a real trace**, following the Langfuse skill's workflow (run
the path, fetch the stored trace, compare with the best-practices page, fix,
repeat). The first real trace showed what tests against an in-memory exporter
could not:

- `search` arrived as a plain `SPAN`: OpenTelemetry keeps 128 attributes per
  span and drops the **oldest** ones first, which were the type keys set at span
  start (50 candidates x 3 attributes). Fix: `SpanLimits(max_span_attributes=512)`.
- The generation had no input: Langfuse does not read OpenInference's
  `llm.input_messages.*`. Fix: also write `langfuse.observation.input` as a
  role-labelled message list, up to 64k characters (the prompt with its context
  is exactly what the model saw).
- Retrieval and verify spans had no input/output, so a reviewer could not see
  what was retrieved. Fix: question in, ranked ids and scores (or the verdict) out.
- The trace name was `answer`, environment `default`, no tags. Fix: stable,
  verb-first trace names on the root (`answer-question`, `eval-question`), and
  environment/release/entry-point tag on every span via the span processor.
  Span names stay: they are also the latency keys in `run.json`.
- API traces had FastAPI's internal dependency/serialization spans and a root
  without input/output. Fix: `operation_spans: False`, question and answer on
  the root span (Langfuse uses the root's I/O as the trace's).

Not done, on purpose: judge verdicts as Langfuse *scores* (needs the Langfuse
API, not OTLP), Langfuse prompt management (prompts are versioned files under
`src/docrag/prompts/` and part of the config hash), and capturing model
reasoning (dropped by design; it is never treated as answer text).

## 7. The eval gate: a quality check that can fail the build

Unit tests prove the code does what it says; they cannot prove the *answers* are
still good. The `eval-gate` CI job runs the real pipeline (ONNX embedder and
reranker, Claude Haiku answers, a DeepSeek judge) on a **fixed** set: 4 small PDFs,
38 questions from the **dev** split, dataset pinned to one Hugging Face revision.
It fails when recall@10 or the judged share of correct answers drops below the
thresholds in `fixtures/ci_gate.yaml`.

Decisions worth defending:

- **dev split, not test**: anything checked on every push becomes a tuning signal.
- **real judge, not a fake**: a fake judge always says "correct", which would make
  the gate a fake. Cost: a few cents per cold run, capped by `DOCRAG_BUDGET_EUR=0.50`
  and by a credit limit on the OpenRouter key.
- **the LLM cache is restored between runs**: its keys are content hashes of
  prompt, context, model and schema, so a changed prompt or retrieval result
  misses the cache and is measured fresh, while unchanged inputs replay. That
  keeps the gate stable when unrelated code changed. Trade-off: provider-side
  model drift is not caught.
- **thresholds with a margin**: with 38 questions one question moves `correct` by
  2.6 points. The gate catches breakage (a retrieval bug, a broken prompt), not
  one-point drifts; the benchmark (`results/`) is where small differences are
  measured, with confidence intervals.
- **fork PRs** get no secrets on GitHub, so they run the retrieval half only and
  the judge check says SKIPPED, never PASSED. On the main repo a missing key fails.
- **a run that evaluated fewer questions than expected fails**: a broken download
  must not pass on three easy questions.

## 8. Tests: a real Qdrant via testcontainers

`tests/e2e` start `qdrant/qdrant:v1.20.0` with testcontainers. *Why not a CI
`services:` container, as before?* The same `uv run pytest` works on a laptop and
in CI, each run gets a fresh server, and the container is removed even when a
test fails. Unit and integration tests run in one `pytest` call, so coverage
(lines **and** branches) counts both. The floor (`fail_under`) sits a few points
under the measured value: dropping tests fails CI, noise does not.

## 9. Deployment: image, compose, Kubernetes

- **Image**: multi-stage (Node for the UI, uv for the venv, slim runtime), non-root
  uid 10001, models not baked in. Its `HEALTHCHECK` uses `/healthz`: Docker restarts
  unhealthy containers, so it must be a liveness check.
- **compose**: Qdrant has a health check (bash opens a TCP socket: the image has no
  curl) and the API starts only when it is healthy (`depends_on: condition`).
- **Kubernetes** (`deploy/k8s`, kustomize): see the table in `docs/deploy.md`. The
  ones to know by heart: startup/liveness/readiness probes as above;
  `replicas: 1` because threads live in memory (a second replica would not know
  them, so scaling needs a shared store first); `maxUnavailable: 0` + `preStop`
  sleep + uvicorn graceful shutdown for rolling updates without dropped requests;
  `readOnlyRootFilesystem` and the `restricted` Pod Security Standard.
- **kind in CI**: the image is built, loaded into a kind cluster and deployed with
  the real manifests and a fake-model config; a script checks rollout, both probes
  and one `/query`. It tests the deployment, not answer quality (that is the gate).

## Interview questions

1. *Your readiness probe checks Qdrant. Why not the liveness probe too?*
2. *A provider returns 429 with `Retry-After: 30`. What does the code do?*
   (Waits 20 s: Retry-After wins but is capped; then retries, within the request
   deadline.)
3. *Why is the CI eval gate on the dev split?*
4. *Why can you restore an LLM cache in CI without measuring stale answers?*
5. *What breaks if you set `replicas: 3`?* (Threads: a follow-up question may land
   on a pod that never saw the thread and gets 404.)
6. *How do you get from a log line to the trace?* (`trace_id` in the JSON line.)
7. *Why `gen_ai.*` and OpenInference attributes on the same span?*
