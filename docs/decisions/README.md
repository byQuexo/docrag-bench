# Architecture decision records

An ADR is a short record of one decision: the context, what was chosen, what
was rejected, and what it costs. Each one fits on a page and points to the
evidence (code, commits, CI runs, result folders), so every "why X and not Y?"
has a written answer that can be checked.

- **Format:** [MADR](https://adr.github.io/madr/)-style, see [`0000-template.md`](0000-template.md).
- **Dates:** the date of the commit that introduced the decision. The git
  history starts with `e1ca9c0` (2026-11-01), which squashes all earlier work;
  ADRs for decisions made before it say so instead of guessing an earlier date.
- **Status:** `Proposed` (open, options only), `Accepted` (in the code),
  `Superseded` (replaced by a later ADR, which is linked). Accepted ADRs are not
  rewritten when the code changes; a new ADR supersedes them.
- **Open rationale:** where no code, commit or doc states why, the ADR says
  **Begründung offen** (rationale open) instead of inventing a reason.
- **Lessons vs ADRs:** [`docs/lessons/`](../lessons/README.md) explain how and
  why things work, at length; ADRs are the lookup entries and link there.

## Index

| # | Decision | Status | Date |
|---|---|---|---|
| | **Retrieval and generation** | | |
| [0001](0001-hybrid-search-qdrant-rerank.md) | Hybrid search (dense + BM25) in Qdrant, then a cross-encoder rerank | Accepted | 2026-11-01 |
| [0002](0002-local-onnx-models.md) | Local ONNX embedder and reranker instead of API or PyTorch models | Accepted | 2026-11-01 |
| [0003](0003-pymupdf-parser.md) | PyMuPDF as the PDF parser, not Docling | Accepted | 2026-11-01 |
| [0004](0004-ocr-separate-pass.md) | OCR as a separate pass: RapidOCR, a vision LLM as the second engine | Accepted | 2026-11-01 |
| [0005](0005-verify-citations-in-code.md) | Citations are verified in code, not by an LLM | Accepted | 2026-11-01 |
| [0006](0006-own-llm-proxy.md) | Our own LLM proxy with a content-addressed cache and a budget cap | Accepted | 2026-11-01 |
| | **Evaluation** | | |
| [0007](0007-document-split-and-statistics.md) | Split by document, page-level metrics, CIs over documents, paired tests | Accepted | 2026-11-01 |
| [0008](0008-llm-judge-with-human-calibration.md) | LLM-as-judge after rules, from another model family, checked against human labels | Accepted | 2026-11-01 |
| [0009](0009-refuse-dirty-benchmark-runs.md) | Benchmark runs refuse to start from a dirty working tree | Accepted | 2026-11-09 |
| [0010](0010-ci-eval-gate.md) | CI eval gate on fixed dev-split questions, thresholds with a margin | Accepted | 2026-11-14 |
| [0011](0011-real-judge-in-ci.md) | A real judge in CI with a spend cap; fork PRs report the judge as skipped | Accepted | 2026-11-14 |
| | **Service and operations** | | |
| [0012](0012-tooling-uv-mypy-precommit.md) | One version source: uv lockfile, mypy strict everywhere, pre-commit through `uv run` | Accepted | 2026-11-01 |
| [0013](0013-sync-qdrant-in-worker-threads.md) | Sync Qdrant client in worker threads, async only for LLM calls | Accepted | 2026-11-01 |
| [0014](0014-retries-and-deadlines.md) | Retries with full jitter and Retry-After; one deadline per request (504) | Accepted | 2026-11-14 |
| [0015](0015-liveness-vs-readiness.md) | `/healthz` checks only the process, `/readyz` checks Qdrant | Accepted | 2026-11-14 |
| [0016](0016-stdlib-json-logs.md) | JSON logs from a stdlib formatter, with the trace id, not structlog | Accepted | 2026-11-14 |
| [0017](0017-opentelemetry-tracing.md) | OpenTelemetry as the tracing layer, query path only | Accepted | 2026-11-01 |
| [0018](0018-langfuse-otlp-exporter.md) | Langfuse as a second OTLP exporter, not the Langfuse SDK | Accepted | 2026-11-14 |
| [0019](0019-testcontainers-for-qdrant.md) | Integration tests start Qdrant with testcontainers | Accepted | 2026-11-14 |
| [0020](0020-container-image.md) | Multi-stage, non-root image; models downloaded at start, not baked in | Accepted | 2026-11-01 |
| [0021](0021-kubernetes-kustomize-kind.md) | Kubernetes manifests with kustomize, tested on kind in CI, one replica | Accepted | 2026-11-14 |
| | **Proposed** | | |
| [0022](0022-shared-thread-store.md) | A shared thread store behind a flag | Proposed | 2026-11-15 |
| [0023](0023-ocr-optional-extra.md) | OCR as an optional extra, to shrink the serving image | Proposed | 2026-11-15 |
| [0024](0024-local-langfuse-in-compose.md) | Langfuse locally in `compose.yaml` | Proposed | 2026-11-15 |
