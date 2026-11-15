# 0001. Hybrid search (dense + BM25) in Qdrant, then a cross-encoder rerank

- **Status:** Accepted
- **Date:** 2026-11-01 (predates the git history; in `e1ca9c0`)

## Context

The questions are about reports, papers and 10-Ks. Dense embeddings match
meaning but miss exact tokens: numbers, names, codes like "10-K" or "Table 3".
BM25 matches tokens but misses paraphrases. The system also needs a way to say
"nothing relevant" instead of always passing the top k chunks to the LLM.

## Decision

Each chunk is stored in Qdrant with a dense vector and a BM25 sparse vector
(`Modifier.IDF`). One server-side query fetches 50 candidates per branch and
fuses them with Reciprocal Rank Fusion, `rrf_k: 60` set explicitly in the
config. A cross-encoder (`gte-reranker-modernbert-base`, ONNX, see
[0002](0002-local-onnx-models.md)) reorders the fused candidates, and its score
against a threshold calibrated on the dev split decides whether anything is
relevant enough to answer.

## Alternatives considered

- **Dense only / BM25 only:** each misses one kind of query (see Context).
- **Weighted sum of scores:** cosine and BM25 scores live on different scales and
  need per-dataset normalisation; RRF uses ranks only and has one parameter.
- **Qdrant's default RRF k=2:** strongly favours each list's top ranks; 60 is the
  value from the RRF paper. Made explicit because the default differs silently.
- **pgvector:** BM25 elsewhere, fusion by hand. **Chroma:** no native sparse/RRF.
  **Pinecone:** managed and paid. **Elasticsearch/OpenSearch:** can do hybrid,
  heavier to run. Qdrant also has an in-process `:memory:` mode for tests.
- **Top-k without a threshold:** always returns k chunks, even for questions the
  document cannot answer.

## Consequences

- **+** One engine, one query for dense, sparse, fusion and filters.
- **+** The rerank threshold lets the system abstain without an LLM call.
- **−** Measured gains are small. Test split recall@10: dense 0.735, BM25 0.716,
  hybrid 0.756, hybrid + rerank 0.752, with overlapping 95% CIs. The README
  lists "reranker: no significant effect, costs latency" as a learning.
- **−** The rerank costs about one second per query on CPU (50 pairs).
- **−** One global threshold ignores per-question difficulty. The rerank gate
  stops only 2.8% of answerable questions; most abstentions come from the LLM.
- **Begründung offen:** why the reranker stays in the serving config although
  it showed no significant recall gain.

## Evidence

- `src/docrag/store/qdrant.py`, `src/docrag/store/retriever.py`,
  `src/docrag/rerank/reranker.py`, `src/docrag/eval/calibrate.py`
- `configs/answer_haiku_vlm_v3.yaml` (`prefetch_k: 50`, `rrf_k: 60`, `rerank_threshold`)
- README "Retrieval table" and "Experiments: Abstention"
- [Lesson 03](../lessons/03_embeddings_and_search.md), [lesson 04](../lessons/04_rerank_and_threshold.md), [interview prep 3, 6–9](../lessons/10_interview_prep.md)
