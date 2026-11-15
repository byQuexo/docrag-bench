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
