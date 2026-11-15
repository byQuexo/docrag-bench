# 0002. Local ONNX embedder and reranker instead of API or PyTorch models

- **Status:** Accepted
- **Date:** 2026-11-01 (predates the git history; in `e1ca9c0`)

## Context

Every query is embedded and its 50 candidates are reranked. Indexing embeds
every chunk of every PDF. Both happen on a laptop (RTX 3070, 8 GB), in CI
runners without a GPU, and in a container.

## Decision

We run both models locally with ONNX Runtime: `gte-small` (384 dim) as the
embedder and `gte-reranker-modernbert-base` (int8) as the reranker, downloaded
from Hugging Face. Tokenisation, pooling and truncation are our own code
(`embed/onnx.py`). The model file and revision are part of the embedding cache
key and of the collection name.

## Alternatives considered

- **sentence-transformers / PyTorch:** several GB of install against ~50 MB for
  ONNX Runtime; the same ONNX files run on CPU, CUDA and DirectML.
- **Embedding/rerank APIs:** a network call and a bill on every query and on
  every re-index; documents leave the machine.
- **`gte-modernbert-base` (768 dim) as embedder:** measured, no gain
  (recall@10 0.754 vs 0.752 for `gte-small`), so the smaller model serves.

## Consequences

- **+** No per-query cost, works offline, small install, same runtime as the
  OCR engine (RapidOCR, [0004](0004-ocr-separate-pass.md)).
- **+** Exact reproducibility: a changed model revision is a different cache
  key and a new collection, never a silent mix of vectors.
- **−** We maintain pooling and truncation ourselves (~40 lines, tested).
- **−** The models (~560 MB) must be downloaded at start-up; see
  [0020](0020-container-image.md).
- **−** GPU (fp16) and CPU (fp32) give slightly different scores: the CI gate
  measured 0.790 recall@10 where a local GPU run gave 0.748.

## Evidence

- `src/docrag/embed/onnx.py`, `src/docrag/embed/registry.py`, `src/docrag/rerank/reranker.py`
- README "Retrieval table" (gte-modernbert-base row)
- [Lesson 03](../lessons/03_embeddings_and_search.md), [lesson 11](../lessons/11_gpu_secrets_security.md), [interview prep 4–5](../lessons/10_interview_prep.md)
