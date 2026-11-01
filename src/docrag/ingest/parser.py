"""PDF -> list[Block]. A Parser Protocol + registry, v1 has one implementation.

Design rules (from the plan / RAG-Anything review):

* The parser fixes ``page`` (1-based) and ``section_path``. Later stages never
  try to reconstruct them from text.
* Tables are normalized to Markdown here, so the chunker doesn't care which
  parser produced them.
* ``version`` goes into the cache key: change the parsing logic -> bump it.

Why PyMuPDF and not Docling (the coach suggestion)? PyMuPDF parses a 50-page PDF
in about a second on CPU without ML models, ships wheels for Windows and finds
ruled tables. Docling gives better layout/table quality but needs PyTorch and
model downloads, and is ~100x slower. Behind the Protocol, adding a
``DoclingParser`` later is one class; the rule is "a second parser only if a
benchmark justifies it". Known trade-off: PyMuPDF is AGPL-3.0 licensed.
"""

import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import pymupdf

from docrag.shared.schema import Block

# Structural typing: any class with these members is a Parser, no inheritance
# needed (unlike llm/base.py, where Provider shares real code).


class Parser(Protocol):
    name: str
    version: str  # goes into the cache key

    def parse(self, pdf: bytes, doc_id: str) -> list[Block]: ...


# --- registry --------------------------------------------------------------

_REGISTRY: dict[str, Callable[[], Parser]] = {}


def register(name: str, factory: Callable[[], Parser]) -> None:
    _REGISTRY[name] = factory


def get_parser(name: str = "pymupdf") -> Parser:
    try:
        return _REGISTRY[name]()
    except KeyError:
        raise ValueError(
            f"Unknown parser {name!r}; known: {sorted(_REGISTRY)}"
        ) from None


# --- PyMuPDF implementation ------------------------------------------------

_CAPTION = re.compile(r"^\s*(table|figure|fig\.|chart|exhibit)\s*[0-9IVX]+", re.I)
_PAGE_NUMBER = re.compile(r"^\s*(page\s*)?\d{1,4}\s*$", re.I)
_BOLD_FLAG = 16  # PyMuPDF span flag bit for bold


@dataclass
class _TextBlock:
    """Internal helper: one PyMuPDF text block, flattened."""

    text: str
    bbox: tuple[float, float, float, float]
    size: float  # dominant font size
    bold: bool
    n_lines: int


def _flatten_text_block(raw: dict[str, Any]) -> _TextBlock | None:
    lines: list[str] = []
    sizes: Counter[float] = Counter()
    bold_chars = 0
    total_chars = 0
    for line in raw["lines"]:
        parts = []
        for span in line["spans"]:
            t = span["text"]
            parts.append(t)
            n = len(t.strip())
            # Weight font sizes by character count, so one big drop-cap letter
            # doesn't turn a paragraph into a heading.
            sizes[round(span["size"], 1)] += n
            total_chars += n
            if span["flags"] & _BOLD_FLAG:
                bold_chars += n
        lines.append("".join(parts))
    text = "\n".join(lines).strip()
    if not text or total_chars == 0:
        return None
    return _TextBlock(
        text=text,
        bbox=tuple(raw["bbox"]),
        size=sizes.most_common(1)[0][0],
        bold=bold_chars > total_chars / 2,
        n_lines=len(lines),
    )


def _overlap_ratio(inner: tuple[float, ...], outer: tuple[float, ...]) -> float:
    """Share of ``inner``'s area that lies inside ``outer`` (0..1)."""
    x0, y0 = max(inner[0], outer[0]), max(inner[1], outer[1])
    x1, y1 = min(inner[2], outer[2]), min(inner[3], outer[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    area = (inner[2] - inner[0]) * (inner[3] - inner[1])
    return ((x1 - x0) * (y1 - y0)) / area if area > 0 else 0.0


class PyMuPdfParser:
    name = "pymupdf"
    version = "1"

    heading_ratio = 1.15  # font size >= body size * ratio -> heading candidate
    max_heading_chars = 120
    margin_ratio = 0.06  # top/bottom 6% of the page = header/footer zone

    def parse(self, pdf: bytes, doc_id: str) -> list[Block]:
        with pymupdf.open(stream=pdf, filetype="pdf") as doc:
            pages = [page.get_text("dict") for page in doc]
            body_size = self._body_font_size(pages)
            heading_levels = self._heading_levels(pages, body_size)

            blocks: list[Block] = []
            stack: list[tuple[int, str]] = []  # (level, heading text)
            order = 0
            for page_index, page in enumerate(doc):
                page_no = page_index + 1  # PDFs are 0-based internally, we are 1-based
                for kind, payload in self._page_items(
                    page, pages[page_index], body_size
                ):
                    if kind == "heading":
                        level = heading_levels[payload.size]
                        # Pop everything at the same or deeper level, then push.
                        while stack and stack[-1][0] >= level:
                            stack.pop()
                        stack.append((level, " ".join(payload.text.split())))
                        continue
                    section = [title for _, title in stack]
                    if kind == "table":
                        md, caption = payload
                        text = f"{caption}\n{md}" if caption else md
                        blocks.append(
                            Block(
                                doc_id=doc_id,
                                type="table",
                                text=text,
                                table_md=md,
                                caption=caption,
                                page=page_no,
                                section_path=section,
                                order=order,
                            )
                        )
                    elif kind == "image":
                        blocks.append(
                            Block(
                                doc_id=doc_id,
                                type="image",
                                text=payload,
                                caption=payload,
                                page=page_no,
                                section_path=section,
                                order=order,
                            )
                        )
                    else:
                        blocks.append(
                            Block(
                                doc_id=doc_id,
                                type="text",
                                text=payload.text,
                                page=page_no,
                                section_path=section,
                                order=order,
                            )
                        )
                    order += 1
        return blocks

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _iter_text_blocks(page_dict: dict[str, Any]) -> list[_TextBlock]:
        out = []
        for raw in page_dict["blocks"]:
            if raw["type"] != 0:  # 0 = text, 1 = image
                continue
            tb = _flatten_text_block(raw)
            if tb is not None:
                out.append(tb)
        return out

    def _body_font_size(self, pages: list[dict[str, Any]]) -> float:
        """Most common font size in the document, weighted by characters."""
        sizes: Counter[float] = Counter()
        for page in pages:
            for tb in self._iter_text_blocks(page):
                sizes[tb.size] += len(tb.text)
        return sizes.most_common(1)[0][0] if sizes else 10.0

    def _is_heading(self, tb: _TextBlock, body_size: float) -> bool:
        return (
            tb.size >= body_size * self.heading_ratio
            and len(tb.text) <= self.max_heading_chars
            and tb.n_lines <= 3
            and not _CAPTION.match(tb.text)
        )

    def _heading_levels(
        self, pages: list[dict[str, Any]], body_size: float
    ) -> dict[float, int]:
        """Map each heading font size to a level: biggest size -> level 0."""
        sizes = {
            tb.size
            for page in pages
            for tb in self._iter_text_blocks(page)
            if self._is_heading(tb, body_size)
        }
        return {size: level for level, size in enumerate(sorted(sizes, reverse=True))}

    def _page_items(
        self, page: pymupdf.Page, page_dict: dict[str, Any], body_size: float
    ) -> list[tuple[str, Any]]:
        """Return the page content in reading order as (kind, payload) tuples."""
        height = page.rect.height
        tables = []
        try:
            for t in page.find_tables().tables:
                md = t.to_markdown().strip()
                # Filter false positives: a "table" needs a header + >= 2 rows.
                if t.row_count >= 2 and t.col_count >= 2 and md:
                    tables.append((tuple(t.bbox), md))
        except Exception:
            # Table detection is a heuristic on vector graphics and can fail on
            # malformed pages. Losing a table beats losing the whole document.
            tables = []

        items: list[tuple[str, Any]] = []
        captions: list[_TextBlock] = []
        placed: set[int] = set()  # indexes of tables already inserted
        for tb in self._iter_text_blocks(page_dict):
            # 1) text inside a table area: replace by the table (once).
            inside = [
                i
                for i, (bbox, _) in enumerate(tables)
                if _overlap_ratio(tb.bbox, bbox) > 0.5
            ]
            if inside:
                for i in inside:
                    if i not in placed:
                        placed.add(i)
                        items.append(("table", i))
                continue
            # 2) running headers/footers and bare page numbers.
            in_margin = tb.bbox[3] < height * self.margin_ratio or tb.bbox[
                1
            ] > height * (1 - self.margin_ratio)
            if _PAGE_NUMBER.match(tb.text) or (in_margin and len(tb.text) < 80):
                continue
            # 3) captions: attached to a table later, or become an image block.
            if _CAPTION.match(tb.text):
                captions.append(tb)
                items.append(("caption", tb))
                continue
            kind = "heading" if self._is_heading(tb, body_size) else "text"
            items.append((kind, tb))
        # Tables whose area contained no text block (rare): append at the end.
        for i in range(len(tables)):
            if i not in placed:
                items.append(("table", i))

        # Attach each caption to the vertically closest table on the page (if close).
        table_caption: dict[int, str] = {}
        image_captions: set[int] = set()
        for cap in captions:
            best, best_dist = None, 80.0  # points; ~3 lines of text
            for i, (bbox, _) in enumerate(tables):
                dist = min(abs(cap.bbox[3] - bbox[1]), abs(cap.bbox[1] - bbox[3]))
                if dist < best_dist and i not in table_caption:
                    best, best_dist = i, dist
            if best is not None and cap.text.lower().startswith("table"):
                table_caption[best] = " ".join(cap.text.split())
            else:
                image_captions.add(id(cap))

        out: list[tuple[str, Any]] = []
        for kind, payload in items:
            if kind == "table":
                bbox, md = tables[payload]
                out.append(("table", (md, table_caption.get(payload))))
            elif kind == "caption":
                if id(payload) in image_captions:
                    # A figure/chart we can't "see": keep its caption as an image
                    # block so the page is still findable by the figure's title.
                    out.append(("image", " ".join(payload.text.split())))
            else:
                out.append((kind, payload))
        return out


register("pymupdf", PyMuPdfParser)
