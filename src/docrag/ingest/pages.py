"""Render a PDF page to PNG, for "click a citation -> see the page".

Rendered on demand (and cheap: ~50 ms per page) instead of pre-rendering ~6000
page images to disk during ingest.
"""

from pathlib import Path

import pymupdf

from docrag.shared.ids import is_valid_doc_id


class UnsafePathError(ValueError):
    """Raised when a requested file would lie outside the allowed directory."""


def safe_pdf_path(pdf_dir: Path, doc_id: str) -> Path:
    """Resolve ``<pdf_dir>/<doc_id>.pdf`` and refuse anything outside ``pdf_dir``.

    ``doc_id`` can come from an HTTP request or from retrieved chunk payloads, so it
    is untrusted: "../../secrets" must never be read (path traversal).
    """
    if not is_valid_doc_id(doc_id):
        raise UnsafePathError(f"invalid doc_id {doc_id!r}")
    base = pdf_dir.resolve()
    path = (base / f"{doc_id}.pdf").resolve()
    # Containment checked after resolve() (also resolves symlinks and "..");
    # never with string startswith().
    if not path.is_relative_to(base):
        raise UnsafePathError(f"{doc_id!r} escapes {base}")
    # Check existence ourselves: pymupdf raises its OWN FileNotFoundError class
    # (not the built-in one), which the API's `except FileNotFoundError` misses.
    if not path.is_file():
        raise FileNotFoundError(f"no PDF for doc_id {doc_id!r}")
    return path


def render_page_png(pdf_path: Path, page: int, zoom: float = 1.5) -> bytes:
    """Render one 1-based page as PNG bytes."""
    with pymupdf.open(pdf_path) as doc:
        if not 1 <= page <= doc.page_count:
            raise IndexError(f"page {page} out of range 1..{doc.page_count}")
        pix = doc[page - 1].get_pixmap(matrix=pymupdf.Matrix(zoom, zoom))
        png: bytes = pix.tobytes("png")
        return png
