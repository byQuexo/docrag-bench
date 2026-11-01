"""list[Block] -> list[Chunk]. Table-aware, page- and heading-preserving.

Rules:

1. Text blocks of the same section are packed together up to ``max_words``.
   A section change always starts a new chunk (chunks don't mix topics).
2. A table is ONE chunk (caption + Markdown) - never cut in the middle of a row.
   Only tables larger than ``max_table_words`` are split by rows, and each piece
   repeats caption + header row so it still makes sense on its own.
3. Figure/chart captions become their own small chunk.
4. Pages are carried through: page_start/page_end come from the blocks.
5. ``text`` stays raw (for display and quote verification); only ``embed_text``
   gets the optional heading prefix (an ablation axis).
"""

import re
from collections import defaultdict
from dataclasses import dataclass, field

from docrag.shared.ids import make_chunk_id
from docrag.shared.schema import Block, BlockType, Chunk, Origin

CHUNKER_VERSION = "1"
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


@dataclass
class ChunkerOptions:
    # ~200 words ~= 260 tokens: fits gte-small's 512-token window with room for the
    # heading prefix, and is small enough that a hit points to a specific passage.
    max_words: int = 200
    max_table_words: int = 350
    prefix_headings: bool = True


def _words(text: str) -> int:
    return len(text.split())


def _split_long_text(text: str, max_words: int) -> list[str]:
    """Split one oversized paragraph at sentence ends (hard cut as a last resort)."""
    pieces: list[str] = []
    current: list[str] = []
    for sentence in _SENTENCE_END.split(text):
        words = sentence.split()
        # A single "sentence" longer than the limit (tables of numbers, lists...).
        while len(words) > max_words:
            if current:
                pieces.append(" ".join(current))
                current = []
            pieces.append(" ".join(words[:max_words]))
            words = words[max_words:]
        if len(current) + len(words) > max_words and current:
            pieces.append(" ".join(current))
            current = []
        current.extend(words)
    if current:
        pieces.append(" ".join(current))
    return pieces


def _split_table(block: Block, max_words: int) -> list[str]:
    """Split a big Markdown table by rows; every piece repeats caption + header."""
    md = block.table_md or block.text
    lines = md.splitlines()
    # Markdown table: line 0 = header, line 1 = |---|---| separator.
    header, rows = lines[:2], lines[2:]
    prefix = ([block.caption] if block.caption else []) + header
    budget = max(max_words - _words("\n".join(prefix)), 20)

    pieces: list[str] = []
    current: list[str] = []
    for row in rows:
        if current and _words("\n".join(current + [row])) > budget:
            pieces.append("\n".join(prefix + current))
            current = []
        current.append(row)
    if current:
        pieces.append("\n".join(prefix + current))
    return pieces


@dataclass
class _Pending:
    """Text being accumulated into the next chunk."""

    doc_id: str = ""
    origin: Origin = "text"
    # (doc_id, section, origin, page-or-None): a change starts a new chunk.
    key: tuple[object, ...] = ()
    texts: list[str] = field(default_factory=list)
    pages: list[int] = field(default_factory=list)
    section: list[str] = field(default_factory=list)

    @property
    def words(self) -> int:
        return sum(_words(t) for t in self.texts)


def chunk_blocks(
    blocks: list[Block], options: ChunkerOptions | None = None
) -> list[Chunk]:
    # LEARN: never use a mutable default like `options=ChunkerOptions()` in the
    # signature: Python evaluates defaults ONCE at definition time, so every call
    # would share the same object. The `None` + "create inside" idiom avoids that.
    opts = options or ChunkerOptions()
    chunks: list[Chunk] = []
    # Per-page counter so IDs read doc:p<page>:c<n> with n restarting on each page.
    next_index: defaultdict[str, int] = defaultdict(int)

    def emit(
        doc_id: str,
        text: str,
        pages: list[int],
        section: list[str],
        kind: BlockType,
        origin: Origin = "text",
    ) -> None:
        # LEARN: a nested function ("closure") can read variables of the enclosing
        # function (chunks, next_index, opts) - like a JS closure. It can mutate
        # the list/dict objects; rebinding a name would need `nonlocal`.
        page_start, page_end = min(pages), max(pages)
        key = f"{doc_id}:{page_start}"
        idx = next_index[key]
        next_index[key] += 1
        prefix = " > ".join(section)
        embed_text = f"{prefix}\n{text}" if opts.prefix_headings and prefix else text
        chunks.append(
            Chunk(
                id=make_chunk_id(doc_id, page_start, idx),
                doc_id=doc_id,
                page_start=page_start,
                page_end=page_end,
                section_path=section,
                type=kind,
                text=text,
                embed_text=embed_text,
                origin=origin,
            )
        )

    pending = _Pending()

    def flush() -> None:
        nonlocal pending  # LEARN: needed because we REBIND `pending` below.
        if pending.texts:
            text = "\n".join(pending.texts)
            emit(
                pending.doc_id,
                text,
                pending.pages,
                pending.section,
                "text",
                pending.origin,
            )
        pending = _Pending()

    # LEARN: sorted(..., key=lambda b: ...) sorts by a computed key; lambda is an
    # anonymous single-expression function, like `b => b.order` in TS.
    for block in sorted(blocks, key=lambda b: (b.doc_id, b.order)):
        # OCR'd pages have no headings (section_path == []), so "same section"
        # would glue consecutive slides into one chunk. For OCR text the page is
        # the unit instead: one slide = one chunk (unless it's very long).
        page_key = block.page if block.origin != "text" else None
        here = (block.doc_id, block.section_path, block.origin, page_key)
        if block.type == "text":
            if pending.texts and pending.key != here:
                flush()
            for piece in _split_long_text(block.text, opts.max_words):
                if pending.texts and pending.words + _words(piece) > opts.max_words:
                    flush()
                pending.doc_id = block.doc_id
                pending.origin = block.origin
                pending.key = here
                pending.texts.append(piece)
                pending.pages.append(block.page)
                pending.section = block.section_path
        elif block.type == "table":
            flush()
            if _words(block.text) <= opts.max_table_words:
                pieces = [block.text]
            else:
                pieces = _split_table(block, opts.max_table_words)
            for piece in pieces:
                emit(block.doc_id, piece, [block.page], block.section_path, "table")
        else:  # image
            flush()
            emit(block.doc_id, block.text, [block.page], block.section_path, "image")
        # Note: we deliberately do NOT flush on a page change. A paragraph that
        # continues on the next page stays one chunk with page_start != page_end.
    flush()
    return chunks
