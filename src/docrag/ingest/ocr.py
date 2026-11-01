"""OCR for pages without a text layer (scanned PDFs, slide decks exported as images).

Found in the first full ingest: 28 of 134 MMLongBench PDFs (1,521 pages, 255 of
1,091 questions) have NO text layer, so the text parser produced nothing for them.

Design:
* OCR is a separate pass AFTER parsing, only for pages where the parser found no
  text but the page has images. Text PDFs pay nothing for it.
* Engines sit behind a small Protocol, like parsers and embedders:
  - ``RapidOcrEngine``: PaddleOCR models run by ONNX Runtime (local, free,
    deterministic, GPU if available). Reads the printed text only.
  - ``VlmOcrEngine`` (``ingest/vlm_ocr.py``): a vision LLM via the proxy. Can
    also describe charts, but may hallucinate text - an eval axis, not a default.
* An engine gets ALL pages of a document at once (``ocr_pages``), so an API-based
  engine can process them concurrently while a local one simply loops.
* Every OCR'd block is marked with the engine's ``origin`` ("ocr" or "vlm") so the
  eval can report it separately: a quote verified against OCR text is only as
  true as the OCR.
* Results are cached per page image (content hash + engine name/version + zoom).
"""

from dataclasses import dataclass
from typing import Any, Protocol

import pymupdf

from docrag.shared.cache import Cache, bytes_hash
from docrag.shared.schema import Block, Origin

# A page "has text" if the text layer has at least this many characters.
MIN_TEXT_CHARS = 20


@dataclass(frozen=True)
class OcrLine:
    text: str
    score: float  # recognition confidence 0..1
    # Top-left corner, used to sort lines into reading order.
    x: float
    y: float


class OcrEngine(Protocol):
    name: str
    version: str  # goes into the cache key
    origin: Origin  # stored on every Block/Chunk this engine produces
    zoom: float  # render scale for page images (2.0 = 144 dpi)

    def ocr_pages(self, pngs: list[bytes]) -> list[str | None]:
        """Text per page image, same order. None = failed (not cached, retried)."""
        ...


class RapidOcrEngine:
    """RapidOCR 3.x (Apache-2.0).

    The detection/recognition models ship INSIDE the pip package and are hash-
    checked when loaded, so pinning the package version pins the models - no
    runtime downloads (same supply-chain rule as our pinned HF revisions).
    """

    name = "rapidocr"
    origin: Origin = "ocr"
    # 2x (144 dpi): small slide text becomes readable; a page image stays ~1-2 MB.
    zoom = 2.0

    @staticmethod
    def version_string() -> str:
        """Engine version without loading the models (for the index fingerprint)."""
        from importlib.metadata import version

        # Package version = model version (the models ship inside the package).
        return f"1:{version('rapidocr')}"

    def __init__(self, min_score: float = 0.5) -> None:
        # Imported here so the module loads without RapidOCR/OpenCV; only creating
        # the engine needs them.
        import rapidocr

        from docrag.embed.onnx import _providers

        self.version = self.version_string()
        use_cuda = _providers()[0] == "CUDAExecutionProvider"
        self._engine = rapidocr.RapidOCR(
            params={
                "EngineConfig.onnxruntime.use_cuda": use_cuda,
                "Global.log_level": "warning",
            }
        )
        self.min_score = min_score

    def ocr(self, png: bytes) -> list[OcrLine]:
        # RapidOCR types its result as a union of 4 classes; a full det+rec call
        # returns this one, which mypy can't narrow.
        result: Any = self._engine(png)
        if result.boxes is None:  # nothing found on the page
            return []
        lines = []
        for box, text, score in zip(
            result.boxes, result.txts, result.scores, strict=True
        ):
            if float(score) < self.min_score or not text.strip():
                continue
            # box = 4 corner points [[x,y], ...]; the first one is top-left.
            lines.append(
                OcrLine(text.strip(), float(score), float(box[0][0]), float(box[0][1]))
            )
        return lines

    def ocr_pages(self, pngs: list[bytes]) -> list[str | None]:
        # Local GPU model: one page after the other is already fast (~0.2 s).
        return [
            "\n".join(line.text for line in reading_order(self.ocr(p))) for p in pngs
        ]


def reading_order(lines: list[OcrLine], line_tolerance: float = 12.0) -> list[OcrLine]:
    """Sort top-to-bottom, and left-to-right within (roughly) the same line.

    y values of boxes on the same visual line differ by a few pixels, so we bucket
    y by ``line_tolerance`` before sorting by x.
    Known limitation: multi-column slides are read row by row, not column by column.
    """
    return sorted(lines, key=lambda line: (round(line.y / line_tolerance), line.x))


def pages_needing_ocr(pdf: bytes) -> list[int]:
    """1-based page numbers with (almost) no text layer but at least one image."""
    pages = []
    with pymupdf.open(stream=pdf, filetype="pdf") as doc:
        for index, page in enumerate(doc):
            if len(page.get_text().strip()) < MIN_TEXT_CHARS and page.get_images():
                pages.append(index + 1)
    return pages


def render_png(pdf: bytes, page: int, zoom: float) -> bytes:
    with pymupdf.open(stream=pdf, filetype="pdf") as doc:
        png: bytes = (
            doc[page - 1].get_pixmap(matrix=pymupdf.Matrix(zoom, zoom)).tobytes("png")
        )
        return png


def ocr_blocks(
    pdf: bytes,
    doc_id: str,
    engine: OcrEngine,
    cache: Cache | None,
    *,
    first_order: int,
) -> list[Block]:
    """OCR every page that needs it and return one text Block per page.

    ``first_order`` continues the reading-order counter after the parser's blocks
    (chunks are sorted by order, and OCR'd pages are image-only pages, so they
    can't interleave with parsed text on the same page).
    """
    pages = pages_needing_ocr(pdf)
    if not pages:
        return []
    pngs = [render_png(pdf, page, engine.zoom) for page in pages]
    keys = [
        f"{engine.name}:{engine.version}:{engine.zoom}:{bytes_hash(png)}"
        for png in pngs
    ]

    texts: list[str | None] = [
        cache.get_json("ocr", key) if cache is not None else None for key in keys
    ]
    missing = [i for i, text in enumerate(texts) if text is None]
    if missing:
        # One call with every uncached page: lets an API engine run them in parallel.
        fresh = engine.ocr_pages([pngs[i] for i in missing])
        for i, text in zip(missing, fresh, strict=True):
            texts[i] = text
            if text is not None and cache is not None:  # never cache failures
                cache.set_json("ocr", keys[i], text)

    blocks: list[Block] = []
    for page, text in zip(pages, texts, strict=True):
        if text and text.strip():
            blocks.append(
                Block(
                    doc_id=doc_id,
                    type="text",
                    text=text,
                    page=page,
                    section_path=[],
                    order=first_order + len(blocks),
                    origin=engine.origin,
                )
            )
    return blocks
