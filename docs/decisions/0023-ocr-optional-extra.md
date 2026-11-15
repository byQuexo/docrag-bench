# 0023. OCR as an optional extra, to shrink the serving image

- **Status:** Proposed
- **Date:** 2026-11-15

## Problem

The serving image holds 948 MB of files (CI, 2026-11-15). OpenCV + RapidOCR
take ~185 MB of that, but OCR only runs at ingest time
([0004](0004-ocr-separate-pass.md)). `rapidocr` is a runtime dependency today.

Two details make this more than moving a line in `pyproject.toml`:

- `rapidocr` is already imported lazily, but `wiring.ocr_version()` reads its
  package version to build the collection fingerprint. A config with
  `ocr: rapidocr` therefore needs the package installed **at serving time** just
  to find its collection. (`ocr: vlm`, the headline config, does not.)
- The API serves page images through PyMuPDF, which stays either way.

## Options

| Option | For | Against |
|---|---|---|
| **`ocr` extra** (`uv sync --extra ocr`), the image installs without it | Smallest change; local dev unchanged with the extra | Serving an `ocr: rapidocr` index needs the version recorded differently (e.g. in the collection's metadata) or the extra installed |
| **Two images: ingest and serve** | Each image has exactly what it needs; ingest becomes a natural Kubernetes Job | Two builds, two tags to keep in sync |
| **Keep as is** | Nothing to change | ~185 MB in every pull and on every node |

## Open questions for the decision

- Is serving an `ocr: rapidocr` index from the image a requirement, or is
  `vlm` (the headline config) enough?
- Should this wait for the ingest Job (`docs/deploy.md`, Before production),
  which would favour two images?
- Measure the result in CI with the existing "Image size" step.

## Evidence

- `pyproject.toml` (`rapidocr==3.10.0`), `src/docrag/wiring.py` (`ocr_version`),
  `src/docrag/ingest/ocr.py`, `Dockerfile`
- [`docs/deploy.md`, The image](../deploy.md), [0020](0020-container-image.md)
