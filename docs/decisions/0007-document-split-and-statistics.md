# 0007. Split by document, page-level metrics, CIs over documents, paired tests

- **Status:** Accepted
- **Date:** 2026-11-01 (split and metrics predate the git history, in `e1ca9c0`; statistics in `acf948f`)

## Context

MMLongBench-Doc labels evidence as pages, not chunks. Questions about the same
PDF are correlated. Claims like "reranking helps" need an uncertainty, and two
configs answer the same questions, so their results are paired.

## Decision

- **Split:** dev/test by a deterministic sha256 of the `doc_id`. Thresholds,
  prompts and `context_k` are chosen on dev; test is the reported number.
- **Metrics at page level:** chunks map to pages, deduplicated in rank order,
  then recall@k, MRR@10, nDCG@10.
- **Uncertainty:** 95% CIs bootstrapped over documents (2,000 reps, seed 0);
  comparisons with an exact paired McNemar test on the disagreements; results
  split by slice (`text`, `table`, `image`, `unanswerable`, `no_text_layer`).

## Alternatives considered

- **Question-level split:** the same PDF ends up in dev and test, and tuning
  leaks into the test number.
- **Python's `hash(doc_id)`:** randomised per process, the split would change
  between runs.
- **Chunk-level metrics:** different chunkers produce different chunks, so they
  could not be compared.
- **Bootstrap over questions / unpaired tests:** ignores the correlation
  within documents and throws away the pairing, giving CIs that are too narrow
  and tests that are too weak.

## Consequences

- **+** Comparable numbers across parsers, chunkers and OCR variants.
- **+** Honest CIs: e.g. the headline correct 0.546 [0.503, 0.591].
- **−** Wide CIs on 735 test questions over 93 documents; small effects are
  not significant (most retrieval ablations).
- **−** The protocol was broken once and is documented as such: `answer_v3` was
  designed from `answer_v2`'s test-split errors, so no held-out claim is made for it.

## Evidence

- `src/docrag/eval/dataset.py`, `src/docrag/eval/metrics.py`, `src/docrag/eval/stats.py`,
  `scripts/reproduce_stats.py` (`7d24767`: recomputes every README statistic)
- README "Results" and "How far to trust these numbers"
- [Lesson 04](../lessons/04_rerank_and_threshold.md), [lesson 08](../lessons/08_eval.md), [interview prep 13–14](../lessons/10_interview_prep.md)
