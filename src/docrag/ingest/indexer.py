"""Index PDFs: parse (parallel, cached) -> chunk -> embed -> upsert into Qdrant."""

import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from docrag.embed.base import EmbeddingService
from docrag.ingest.chunker import ChunkerOptions, chunk_blocks
from docrag.ingest.ocr import OcrEngine, ocr_blocks
from docrag.ingest.parser import get_parser
from docrag.ingest.pipeline import parse_cache_key, parse_pdf
from docrag.shared.cache import Cache
from docrag.shared.schema import Block
from docrag.store.qdrant import QdrantStore

log = logging.getLogger(__name__)


def _parse_worker(args: tuple[Path, str]) -> tuple[str, list[Block]]:
    # LEARN: must be a top-level function: ProcessPoolExecutor sends work to other
    # processes by pickling it, and closures/lambdas can't be pickled.
    path, doc_id = args
    return doc_id, parse_pdf(path, doc_id, get_parser(), cache=None)


def index_documents(
    pdfs: dict[str, Path],
    store: QdrantStore,
    embeddings: EmbeddingService,
    *,
    prefix_headings: bool,
    cache: Cache | None,
    workers: int = 3,
    ocr: OcrEngine | None = None,
) -> int:
    """Index every doc not yet in ``store``. Returns the number of chunks added.

    With ``ocr``, pages that have no text layer but contain images are OCR'd in
    this (main) process, after parsing: the GPU model is loaded once, not once
    per worker process.
    """
    todo = {d: p for d, p in pdfs.items() if store.count(d) == 0}
    if not todo:
        return 0
    log.info("indexing %d documents into %s", len(todo), store.collection)

    # 1) Parse what isn't cached, in parallel processes.
    # LEARN: Why processes and not threads? Parsing is CPU-bound Python code, and
    # CPython's GIL lets only one thread run Python bytecode at a time. Separate
    # processes each have their own interpreter + GIL, so 4 workers ~ 4x faster.
    # (asyncio wouldn't help either: it's for waiting on I/O, not for computing.)
    parser = get_parser()
    blocks_by_doc: dict[str, list[Block]] = {}
    uncached: list[tuple[Path, str]] = []
    for doc_id, path in todo.items():
        if cache is not None and _cached(cache, path, doc_id):
            blocks_by_doc[doc_id] = parse_pdf(path, doc_id, parser, cache)
        else:
            uncached.append((path, doc_id))
    if uncached:
        paths = {doc_id: path for path, doc_id in uncached}

        def done(doc_id: str, blocks: list[Block]) -> None:
            # Cache EACH document as soon as it's parsed, so an interrupted run
            # (Ctrl+C, crash) only loses the PDFs that were still in progress.
            blocks_by_doc[doc_id] = blocks
            if cache is not None:
                _store_parse(cache, paths[doc_id], doc_id, blocks)
            log.info("parsed %s (%d blocks)", doc_id, len(blocks))

        if workers > 1 and len(uncached) > 1:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                # LEARN: as_completed yields futures in FINISH order, so a fast
                # document isn't stuck waiting behind a slow 300-page one
                # (pool.map would yield in INPUT order).
                futures = [pool.submit(_parse_worker, args) for args in uncached]
                for future in as_completed(futures):
                    done(*future.result())
        else:
            for args in uncached:
                done(*_parse_worker(args))

    # 2) Chunk, embed, upsert - one document at a time to bound memory.
    added = 0
    for doc_id in sorted(blocks_by_doc):
        blocks = blocks_by_doc[doc_id]
        if ocr is not None:
            ocr_found = ocr_blocks(
                todo[doc_id].read_bytes(),
                doc_id,
                ocr,
                cache,
                first_order=len(blocks),
            )
            if ocr_found:
                log.info("%s: OCR'd %d pages (%s)", doc_id, len(ocr_found), ocr.name)
            blocks = blocks + ocr_found
        chunks = chunk_blocks(blocks, ChunkerOptions(prefix_headings=prefix_headings))
        if not chunks:
            log.warning(
                "%s produced no chunks (scanned PDF without text layer?)", doc_id
            )
            continue
        vectors = embeddings.embed([c.embed_text for c in chunks])
        store.upsert(chunks, vectors)
        added += len(chunks)
        log.info("indexed %s: %d chunks", doc_id, len(chunks))
    return added


def _key(path: Path, doc_id: str) -> str:
    return parse_cache_key(path.read_bytes(), get_parser(), doc_id)


def _cached(cache: Cache, path: Path, doc_id: str) -> bool:
    return cache.get_bytes("parse", _key(path, doc_id)) is not None


def _store_parse(cache: Cache, path: Path, doc_id: str, blocks: list[Block]) -> None:
    cache.set_json(
        "parse", _key(path, doc_id), [b.model_dump(mode="json") for b in blocks]
    )
