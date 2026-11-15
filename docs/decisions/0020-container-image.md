# 0020. Multi-stage, non-root image; models downloaded at start, not baked in

- **Status:** Accepted
- **Date:** 2026-11-01 (multi-stage, non-root, in `e1ca9c0`); hardened 2026-11-14 (`7fe38a9`, `7ba8a30`)

## Context

The image runs locally in compose, in kind, and in a cluster that enforces the
`restricted` Pod Security Standard. The ONNX models are ~560 MB. Build tools
and the Node toolchain for the web UI are not needed at runtime.

## Decision

Three stages: Node builds the UI, a uv stage builds the venv from `uv.lock`,
and the final `python:3.12-slim` stage gets only the UI files and `.venv`. It
runs as uid 10001, writes only to `/app/models` and `/app/.cache`, logs JSON,
and has a `HEALTHCHECK` on `/healthz`. Models are downloaded into
`/app/models` on first start. scikit-learn and SciPy were dropped from the
runtime dependencies (`cohen_kappa` is hand-written now).

## Alternatives considered

- **Single stage:** compilers, Node and caches end up in the runtime image.
- **Models baked into the image:** +560 MB per image and slower rebuilds, in
  exchange for a fast first start.
- **Root user:** rejected by the `restricted` policy and a larger blast radius.

## Consequences

- **+** Small attack surface, runs under `readOnlyRootFilesystem`.
- **+** Rebuilds are fast and the image does not change when a model is re-pinned.
- **−** Still not small: 948 MB of files in CI (365 MB as reported by
  `docker image inspect`, the compressed size), mostly polars, OpenCV + RapidOCR
  and PyMuPDF. Next step: [0023](0023-ocr-optional-extra.md).
- **−** Slow first start, and in Kubernetes the models sit in an `emptyDir`, so
  every new pod downloads them again.

## Evidence

- `Dockerfile`, `.github/workflows/ci.yml` (step "Image size"), `src/docrag/eval/agreement.py`
- [`docs/deploy.md`, The image](../deploy.md), [lesson 14, section 9](../lessons/14_operability.md)
