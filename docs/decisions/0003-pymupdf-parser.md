# 0003. PyMuPDF as the PDF parser, not Docling

- **Status:** Accepted
- **Date:** 2026-11-01 (predates the git history; in `e1ca9c0`)

## Context

The benchmark parses 134 PDFs, some with hundreds of pages, and must re-parse
them whenever the chunker changes. The PDFs are mostly born-digital, so their
text layer is exact.

## Decision

We parse with PyMuPDF into typed `Block`s (page, section path, text or table)
behind a `Parser` Protocol. Tables become one chunk each. Parsing runs in
worker processes and is cached by content.

## Alternatives considered

- **Docling:** better layout and table structure, but pulls in PyTorch and model
  downloads and is much slower. It is one class away behind the Protocol; the
  rule is to add it only if the table slice shows parsing is the bottleneck.
- **Unstructured / an OCR pipeline for everything:** OCR adds errors and cost on
  PDFs whose text layer is already exact (scans get OCR separately, [0004](0004-ocr-separate-pass.md)).

## Consequences

- **+** About 50 pages per second on CPU, no ML models, Windows wheels.
- **+** The parser is a pure function of the PDF bytes, so its cache stays valid.
- **−** PyMuPDF is AGPL-3.0, which matters for a closed-source product.
- **−** Table detection dominates parse time (~115 s for a 130-page 10-K), and
  the heading heuristic is noisy on marketing-style PDFs.
- **−** Charts are invisible except for captions and labels.

## Evidence

- `src/docrag/ingest/parser.py`, `src/docrag/ingest/chunker.py`
- [Lesson 02](../lessons/02_ingest.md), [interview prep 1, 27](../lessons/10_interview_prep.md)
