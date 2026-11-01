"""PDF file -> Blocks (cached) -> Chunks."""

import logging
from pathlib import Path

from docrag.ingest.chunker import ChunkerOptions, chunk_blocks
from docrag.ingest.parser import Parser, get_parser
from docrag.shared import timing
from docrag.shared.cache import Cache, bytes_hash
from docrag.shared.schema import Block, Chunk
from docrag.shared.timing import timed

log = logging.getLogger(__name__)


def parse_cache_key(data: bytes, parser: Parser, doc_id: str) -> str:
    # Content hash + parser name/version (+ doc_id, which is stored in every Block).
    # NOT the path or mtime: renaming a file must not invalidate, editing must.
    return f"{parser.name}:{parser.version}:{bytes_hash(data)}:{doc_id}"


@timed("parse")
def parse_pdf(
    path: Path, doc_id: str, parser: Parser, cache: Cache | None
) -> list[Block]:
    data = path.read_bytes()
    key = parse_cache_key(data, parser, doc_id)
    if cache is not None:
        cached = cache.get_json("parse", key)
        if cached is not None:
            timing.cache_hit()
            return [Block.model_validate(b) for b in cached]
    blocks = parser.parse(data, doc_id)
    if cache is not None:
        cache.set_json("parse", key, [b.model_dump(mode="json") for b in blocks])
    return blocks


def ingest_pdf(
    path: Path,
    doc_id: str | None = None,
    *,
    prefix_headings: bool = True,
    parser_name: str = "pymupdf",
    cache: Cache | None = None,
) -> list[Chunk]:
    doc_id = doc_id or path.stem
    blocks = parse_pdf(path, doc_id, get_parser(parser_name), cache)
    chunks = chunk_blocks(blocks, ChunkerOptions(prefix_headings=prefix_headings))
    log.info("%s: %d blocks -> %d chunks", doc_id, len(blocks), len(chunks))
    return chunks
