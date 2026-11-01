"""Composition root: build the object graph from Settings + ExperimentConfig.

Every other module receives its dependencies as arguments ("dependency
injection" without a framework). Only this file knows how to construct
everything, so tests can build the same graph with fakes instead.
"""

import logging

from docrag.answer.service import AnswerService
from docrag.embed.base import Embedder, EmbeddingService
from docrag.embed.registry import make_embedder
from docrag.ingest.chunker import CHUNKER_VERSION
from docrag.ingest.ocr import OcrEngine
from docrag.ingest.parser import get_parser
from docrag.llm.proxy import LLMProxy
from docrag.rerank.reranker import make_reranker
from docrag.shared.cache import Cache, stable_hash
from docrag.shared.config import ExperimentConfig
from docrag.shared.settings import Settings
from docrag.store.qdrant import QdrantStore, collection_name, make_client
from docrag.store.retriever import Retriever

log = logging.getLogger(__name__)


def ocr_version(config: ExperimentConfig) -> str:
    """Version string of the configured OCR engine, without loading it."""
    match config.ocr:
        case "none":
            return "none"
        case "rapidocr":
            from docrag.ingest.ocr import RapidOcrEngine

            return f"rapidocr:{RapidOcrEngine.version_string()}"
        case "vlm":
            from docrag.ingest.vlm_ocr import VlmOcrEngine

            return f"vlm:{VlmOcrEngine.version_string(config.ocr_model or '')}"


def index_components(config: ExperimentConfig, embedder: Embedder) -> dict[str, str]:
    """Everything that decides which chunks and vectors end up in the index.

    Bump PARSER/CHUNKER versions (or change the embedder/OCR) and the
    fingerprint changes -> a new collection -> a fresh index. Parse, OCR and
    embedding caches still make the re-index cheap for unchanged parts.
    """
    parser = get_parser()
    return {
        "parser": f"{parser.name}:{parser.version}",
        "chunker": CHUNKER_VERSION,
        "prefix_headings": str(config.prefix_headings),
        "embedder": f"{embedder.name}:{embedder.version}",
        "ocr": ocr_version(config),
    }


def build_store(
    settings: Settings, config: ExperimentConfig, embedder: Embedder
) -> QdrantStore:
    ocr = config.ocr if config.ocr != "vlm" else f"vlm-{config.ocr_model}"
    components = index_components(config, embedder)
    name = collection_name(
        config.embedder,
        embedder.dim,
        config.prefix_headings,
        ocr,
        fingerprint=stable_hash(components)[:8],
    )
    log.info("collection %s <- %s", name, components)
    store = QdrantStore(make_client(settings.qdrant_location), name, embedder.dim)
    store.ensure_collection()
    return store


def build_ocr(
    settings: Settings, config: ExperimentConfig, cache: Cache | None
) -> OcrEngine | None:
    """The OCR engine for pages without a text layer (None = no OCR)."""
    match config.ocr:
        case "none":
            return None
        case "rapidocr":
            from docrag.ingest.ocr import RapidOcrEngine

            return RapidOcrEngine()
        case "vlm":
            from docrag.ingest.vlm_ocr import VlmOcrEngine

            if config.ocr_model is None:
                raise ValueError("ocr: vlm needs ocr_model (a 'provider:model' route)")
            return VlmOcrEngine(
                LLMProxy.from_settings(settings, cache),
                config.ocr_model,
                temperature=config.ocr_temperature,
            )


def build_retriever(
    settings: Settings, config: ExperimentConfig, cache: Cache | None
) -> Retriever:
    embedder = make_embedder(config.embedder, settings.models_dir)
    embeddings = EmbeddingService(embedder, cache, settings.embed_batch_size)
    reranker = (
        make_reranker(config.reranker, settings.models_dir)
        if config.retrieval == "hybrid_rerank"
        else None
    )
    return Retriever(
        embeddings, build_store(settings, config, embedder), config, reranker
    )


def build_answer_service(
    settings: Settings, config: ExperimentConfig, cache: Cache | None
) -> AnswerService:
    if config.llm is None:
        raise ValueError(f"config {config.name!r} has no `llm` route")
    return AnswerService(
        build_retriever(settings, config, cache),
        LLMProxy.from_settings(settings, cache),
        config.llm,
    )
