"""The Embedder interface and the shared service around it.

Split of responsibilities (from the plan):

* An *Embedder backend* only knows how to turn a list of texts into raw vectors.
  Adding a backend = one small class with ``embed()``.
* ``EmbeddingService`` does everything else exactly once for all backends:
  batching, caching by (model, hash(text)), and L2 normalization.
"""

from typing import Protocol

import numpy as np
import numpy.typing as npt

from docrag.shared import timing, tracing
from docrag.shared.cache import Cache, stable_hash
from docrag.shared.timing import timed

# LEARN: a type alias for "2-D float32 numpy array". numpy arrays are the Python
# equivalent of a typed contiguous buffer (Rust Vec<f32> with a shape on top).
# Vector math on them runs in C, which is why ML code in Python is fast enough.
Vectors = npt.NDArray[np.float32]


class Embedder(Protocol):
    name: str  # e.g. "gte-small"; part of cache key and collection name
    version: str  # bump when pooling/truncation changes -> cache invalidates
    dim: int

    def embed(self, texts: list[str]) -> Vectors: ...  # raw, shape (len(texts), dim)


def l2_normalize(vectors: Vectors) -> Vectors:
    """Scale each row to length 1, so dot product == cosine similarity."""
    # LEARN: numpy "broadcasting": norms has shape (n, 1), vectors (n, dim); the
    # division applies each row's norm to every column of that row. No loops.
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return (vectors / np.clip(norms, 1e-12, None)).astype(np.float32)


class EmbeddingService:
    def __init__(
        self, embedder: Embedder, cache: Cache | None = None, batch_size: int = 32
    ):
        self.embedder = embedder
        self.cache = cache
        self.batch_size = batch_size

    @property
    def dim(self) -> int:
        return self.embedder.dim

    def _key(self, text: str) -> str:
        return stable_hash(self.embedder.name, self.embedder.version, text)

    @timed("embed", kind=tracing.EMBEDDING)
    def embed(self, texts: list[str]) -> Vectors:
        """Embed texts with cache + batching. Returns L2-normalized float32 rows."""
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        missing: list[int] = []  # positions not found in the cache
        for i, text in enumerate(texts):
            raw = self.cache.get_bytes("embed", self._key(text)) if self.cache else None
            if raw is None:
                missing.append(i)
            else:
                # LEARN: np.frombuffer views raw bytes as an array without copying.
                out[i] = np.frombuffer(raw, dtype=np.float32)

        if not missing:
            timing.cache_hit()  # no model call at all
        for start in range(0, len(missing), self.batch_size):
            idx = missing[start : start + self.batch_size]
            vecs = l2_normalize(self.embedder.embed([texts[i] for i in idx]))
            # LEARN: "fancy indexing": out[[3, 7, 9]] = vecs writes 3 rows at once.
            out[idx] = vecs
            if self.cache:
                for i, vec in zip(idx, vecs, strict=True):
                    # LEARN: zip(..., strict=True) raises if lengths differ instead
                    # of silently stopping at the shorter one (ruff rule B905).
                    self.cache.set_bytes("embed", self._key(texts[i]), vec.tobytes())
        tracing.annotate(
            {
                "embedding.model_name": self.embedder.name,
                "docrag.n_texts": len(texts),
                "docrag.cache_misses": len(missing),
            }
        )
        return out

    def embed_query(self, text: str) -> Vectors:
        """One text -> one 1-D vector (the first and only row)."""
        row: Vectors = self.embed([text])[0]
        return row
