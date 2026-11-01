"""Qdrant: one collection per (embedding model, dim, prefix flag), hybrid search + RRF.

Each point stores:
* ``dense``: the embedding of ``chunk.embed_text`` (cosine distance)
* ``bm25``:  a sparse BM25 vector (IDF applied by Qdrant at query time)
* payload:   the full Chunk as JSON, so a hit can be shown/cited without a 2nd DB.

Why one collection per model? Vectors of different models live in different
spaces (and even different dims). Mixing them in one collection is meaningless;
naming the collection after model+dim makes accidental mixing impossible.
"""

import logging
import re

from qdrant_client import QdrantClient, models

from docrag.embed.base import Vectors
from docrag.shared import tracing
from docrag.shared.config import RetrievalMode
from docrag.shared.ids import chunk_point_id
from docrag.shared.schema import Chunk, Hit
from docrag.shared.timing import timed
from docrag.store.bm25 import doc_sparse_vector, query_sparse_vector

log = logging.getLogger(__name__)

DENSE = "dense"
SPARSE = "bm25"


def make_client(location: str) -> QdrantClient:
    """':memory:' (tests), 'http(s)://...' (server) or a folder path (embedded)."""
    if location == ":memory:":
        return QdrantClient(":memory:")
    if location.startswith(("http://", "https://")):
        return QdrantClient(url=location, timeout=60)
    # Embedded mode: Qdrant's Python implementation persists to a folder.
    # Fine for development; slow beyond ~20k points (brute-force search).
    return QdrantClient(path=location)


def collection_name(
    embedder: str,
    dim: int,
    prefix_headings: bool,
    ocr: str = "none",
    fingerprint: str | None = None,
) -> str:
    safe = re.sub(r"[^A-Za-z0-9_-]", "-", embedder)
    name = f"chunks__{safe}__{dim}__{'pfx' if prefix_headings else 'raw'}"
    # OCR changes which chunks exist, so OCR variants get their own collection.
    # A VLM variant includes its model ("vlm-openrouter:anthropic/..."), so two
    # vision models never share a collection; sanitized like the embedder name.
    if ocr != "none":
        name = f"{name}__ocr-{re.sub(r'[^A-Za-z0-9_.-]', '-', ocr)}"
    # The fingerprint hashes the versions of everything that produced the chunks
    # (parser, chunker, embedder, OCR engine; see wiring.index_components). The
    # indexer skips documents that already have chunks, so without it, changing
    # the chunker would silently keep measuring the OLD index.
    return f"{name}__v{fingerprint}" if fingerprint else name


class QdrantStore:
    def __init__(self, client: QdrantClient, collection: str, dim: int) -> None:
        self.client = client
        self.collection = collection
        self.dim = dim

    def ensure_collection(self) -> None:
        if self.client.collection_exists(self.collection):
            return
        self.client.create_collection(
            self.collection,
            vectors_config={
                DENSE: models.VectorParams(
                    size=self.dim, distance=models.Distance.COSINE
                )
            },
            sparse_vectors_config={
                SPARSE: models.SparseVectorParams(modifier=models.Modifier.IDF)
            },
        )
        # Keyword index on doc_id: makes "search only in this document" a cheap
        # filter instead of a full scan. (Ignored by the in-memory mode.)
        self.client.create_payload_index(
            self.collection, "doc_id", field_schema=models.PayloadSchemaType.KEYWORD
        )
        log.info("created collection %s", self.collection)

    def count(self, doc_id: str | None = None) -> int:
        flt = _doc_filter([doc_id]) if doc_id else None
        return self.client.count(self.collection, count_filter=flt, exact=True).count

    @timed("upsert")
    def upsert(
        self, chunks: list[Chunk], vectors: Vectors, batch_size: int = 256
    ) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        points = [
            models.PointStruct(
                id=chunk_point_id(chunk.id),
                vector={
                    DENSE: vec.tolist(),
                    SPARSE: doc_sparse_vector(chunk.embed_text),
                },
                payload=chunk.model_dump(mode="json"),
            )
            for chunk, vec in zip(chunks, vectors, strict=True)
        ]
        for start in range(0, len(points), batch_size):
            self.client.upsert(self.collection, points[start : start + batch_size])

    @timed("search", kind=tracing.RETRIEVER)
    def search(
        self,
        *,
        query_vector: Vectors | None,
        query_text: str,
        mode: RetrievalMode,
        limit: int = 10,
        prefetch_k: int = 50,
        rrf_k: int = 60,
        doc_ids: list[str] | None = None,
    ) -> list[Hit]:
        flt = _doc_filter(doc_ids) if doc_ids else None
        sparse = query_sparse_vector(query_text)
        has_terms = bool(sparse.indices)

        if mode == "dense" or (mode != "bm25" and not has_terms):
            # Also the fallback for hybrid when the query has no usable keywords.
            res = self.client.query_points(
                self.collection,
                query=_dense(query_vector),
                using=DENSE,
                query_filter=flt,
                limit=limit,
                with_payload=True,
            )
            source = "dense"
        elif mode == "bm25":
            if not has_terms:
                return []
            res = self.client.query_points(
                self.collection,
                query=sparse,
                using=SPARSE,
                query_filter=flt,
                limit=limit,
                with_payload=True,
            )
            source = "bm25"
        else:  # "hybrid" and "hybrid_rerank" retrieve the same way
            # Two prefetch branches run inside Qdrant, then Reciprocal Rank Fusion:
            #   rrf(d) = sum over branches 1 / (k + rank_in_branch(d))
            # RRF only uses RANKS, so cosine scores (0..1) and BM25 scores (0..30+)
            # never need to be calibrated against each other.
            res = self.client.query_points(
                self.collection,
                prefetch=[
                    models.Prefetch(
                        query=_dense(query_vector),
                        using=DENSE,
                        filter=flt,
                        limit=prefetch_k,
                    ),
                    models.Prefetch(
                        query=sparse, using=SPARSE, filter=flt, limit=prefetch_k
                    ),
                ],
                query=models.RrfQuery(rrf=models.Rrf(k=rrf_k)),
                limit=limit,
                with_payload=True,
            )
            source = "hybrid"
        hits = [
            Hit(chunk=Chunk.model_validate(p.payload), score=p.score, source=source)
            for p in res.points
        ]
        # First-stage candidates (short content: there can be 50 of them).
        tracing.annotate(
            {
                "docrag.mode": source,
                "docrag.limit": limit,
                "docrag.doc_filter": ",".join(doc_ids) if doc_ids else None,
            }
        )
        tracing.annotate_documents(
            "retrieval.documents", tracing.hit_docs(hits), max_content=200
        )
        return hits


def _dense(vec: Vectors | None) -> list[float]:
    if vec is None:
        raise ValueError("this retrieval mode needs a query embedding")
    return [float(x) for x in vec]


def _doc_filter(doc_ids: list[str]) -> models.Filter:
    return models.Filter(
        must=[models.FieldCondition(key="doc_id", match=models.MatchAny(any=doc_ids))]
    )
