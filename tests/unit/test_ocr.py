"""OCR for pages without a text layer (ingest/ocr.py) and how its text is chunked."""

from pathlib import Path

import pytest

from docrag.ingest.chunker import chunk_blocks
from docrag.ingest.ocr import OcrLine, ocr_blocks, pages_needing_ocr, reading_order
from docrag.shared.cache import Cache
from docrag.store.qdrant import collection_name

SCANNED = Path(__file__).resolve().parents[2] / "fixtures" / "tiny_scanned.pdf"


class FakeOcrEngine:
    """Returns fixed text per page and counts pages (to prove caching works)."""

    name = "fake-ocr"
    version = "1"
    origin = "ocr"
    zoom = 1.0

    def __init__(self) -> None:
        self.calls = 0

    def ocr_pages(self, pngs: list[bytes]) -> list[str | None]:
        out: list[str | None] = []
        for _ in pngs:
            self.calls += 1
            # Out of order on purpose: reading_order() must put "Slide n" first.
            lines = [
                OcrLine("Europe 41.0 45.5", 0.99, x=10, y=200),
                OcrLine(f"Slide {self.calls}", 0.98, x=10, y=20),
            ]
            out.append("\n".join(line.text for line in reading_order(lines)))
        return out


def test_pages_needing_ocr(tiny_pdf: Path) -> None:
    assert pages_needing_ocr(SCANNED.read_bytes()) == [1, 2, 3]
    assert pages_needing_ocr(tiny_pdf.read_bytes()) == []  # has a text layer


def test_reading_order_is_top_to_bottom_then_left_to_right() -> None:
    lines = [
        OcrLine("right", 1, x=300, y=101),  # same visual line as "left"
        OcrLine("bottom", 1, x=0, y=300),
        OcrLine("left", 1, x=0, y=99),
    ]
    assert [line.text for line in reading_order(lines)] == ["left", "right", "bottom"]


def test_ocr_blocks_one_block_per_page_marked_ocr() -> None:
    engine = FakeOcrEngine()
    blocks = ocr_blocks(SCANNED.read_bytes(), "scan", engine, None, first_order=5)
    assert [b.page for b in blocks] == [1, 2, 3]
    assert all(b.origin == "ocr" and b.type == "text" for b in blocks)
    assert [b.order for b in blocks] == [5, 6, 7]  # continues after parsed blocks
    assert blocks[0].text.splitlines()[0] == "Slide 1"  # reading order applied


def test_ocr_results_are_cached_per_page_image() -> None:
    cache = Cache(":memory:")
    engine = FakeOcrEngine()
    first = ocr_blocks(SCANNED.read_bytes(), "scan", engine, cache, first_order=0)
    second = ocr_blocks(SCANNED.read_bytes(), "scan", engine, cache, first_order=0)
    assert engine.calls == 3  # second run: all 3 pages from the cache
    assert [b.text for b in first] == [b.text for b in second]


def test_ocr_text_is_chunked_per_page_never_glued_across_slides() -> None:
    blocks = ocr_blocks(
        SCANNED.read_bytes(), "scan", FakeOcrEngine(), None, first_order=0
    )
    chunks = chunk_blocks(blocks)
    assert [(c.page_start, c.page_end) for c in chunks] == [(1, 1), (2, 2), (3, 3)]
    assert all(c.origin == "ocr" for c in chunks)


def test_ocr_variant_gets_its_own_collection() -> None:
    plain = collection_name("onnx:gte-small", 384, True)
    assert plain == "chunks__onnx-gte-small__384__pfx"  # unchanged: old data valid
    assert collection_name("onnx:gte-small", 384, True, "rapidocr") == (
        plain + "__ocr-rapidocr"
    )


def test_rapidocr_reads_the_scanned_table() -> None:
    """The real engine on the image-only fixture (CPU in CI, ~2 s)."""
    pytest.importorskip("rapidocr")
    from docrag.ingest.ocr import RapidOcrEngine

    blocks = ocr_blocks(
        SCANNED.read_bytes(), "scan", RapidOcrEngine(), None, first_order=0
    )
    page2 = next(b for b in blocks if b.page == 2).text
    assert "Europe" in page2
    assert "45.5" in page2
