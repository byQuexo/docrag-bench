# 0004. OCR as a separate pass: RapidOCR, a vision LLM as the second engine

- **Status:** Accepted
- **Date:** 2026-11-01 (predates the git history; in `e1ca9c0`)

## Context

A fifth of the benchmark scored zero: 23 image-only PDFs (mostly slide decks)
have no text layer, so the parser produced no chunks for them. Recall@10 on
those questions was 0.000.

## Decision

OCR runs as a second pass at index time, only on pages with (almost) no text
but at least one image. Two engines, selected per config (`ocr: rapidocr | vlm`):
RapidOCR (PaddleOCR models on ONNX Runtime, local) and a vision LLM (Claude
Haiku via OpenRouter) that also describes charts. Every chunk records its
`origin`, and each OCR variant gets its own Qdrant collection.

## Alternatives considered

- **OCR inside the parser:** the parser runs in three worker processes; loading
  the OCR model on the GPU in each would load it three times and fight over VRAM.
  It would also break the parser's cache.
- **Tesseract:** a separate system install on Windows, in Docker and in CI;
  weaker on slides.
- **Docling OCR / DeepSeek-OCR:** PyTorch (GBs), tight on 8 GB VRAM.

## Consequences

- **+** Scanned PDFs recall@10: 0.000 → 0.776 (RapidOCR) → 0.834 (vision LLM);
  text-layer PDFs are unchanged (0.752 / 0.753 / 0.755). The biggest single
  gain in the project.
- **+** Normal PDFs pay nothing: the check is a text-length test per page.
- **−** The vision LLM costs ~0.0003 EUR per page (0.42 EUR for 1,390 pages)
  and sends pages to a third party; a hallucinated transcription would be
  invisible to the judge, which only sees chunk text.
- **−** OpenCV and RapidOCR add ~185 MB to the serving image, although OCR is
  only needed at ingest time. See [0023](0023-ocr-optional-extra.md).

## Evidence

- `src/docrag/ingest/ocr.py`, `src/docrag/ingest/vlm_ocr.py`, `src/docrag/ingest/pipeline.py`
- `configs/answer_haiku_vlm_v3.yaml` (`ocr: vlm`), README "Experiments: OCR"
- [Lesson 12](../lessons/12_ocr.md)
