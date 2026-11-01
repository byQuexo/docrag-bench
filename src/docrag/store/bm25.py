"""BM25 as Qdrant sparse vectors.

Correction to the plan: Qdrant's *full-text payload index* only FILTERS ("contains
these words: yes/no"), it does not RANK. Ranked keyword search in Qdrant is done
with sparse vectors + ``Modifier.IDF``. BM25 splits cleanly into two halves:

    score(q, d) = sum over query terms t of  IDF(t) * W(t, d)

    W(t, d) = tf * (k1 + 1) / (tf + k1 * (1 - b + b * |d| / avgdl))

    IDF(t)  -> computed by Qdrant at query time
    W(t, d) -> computed by us, per document, at index time

* Document side (here): the TF-saturation + length-normalization part.
* IDF: needs corpus statistics that change as documents are added, so Qdrant
  computes it at query time (``modifier=IDF`` on the sparse vector config).
* Query side: weight 1 per unique query term.

Term IDs: crc32(term). Collisions are possible but rare (2^32 space) - a known,
accepted inaccuracy, same trick as Qdrant's own BM25 / fastembed.
"""

import zlib
from collections import Counter

from qdrant_client import models

from docrag.shared.text import tokenize

K1 = 1.2  # TF saturation: the 10th occurrence adds much less than the 2nd
B = 0.75  # length normalization strength (0 = none, 1 = full)
# Average document length (in tokens after stopword removal). A fixed constant
# instead of a live corpus statistic, so a chunk's vector never depends on which
# other documents were ingested in the same run (Qdrant's own BM25 does the same).
# Roughly the mean of our chunks; measured with average_length() on the corpus.
AVGDL = 90.0


def term_id(term: str) -> int:
    return zlib.crc32(term.encode("utf-8"))


def doc_sparse_vector(text: str, avgdl: float = AVGDL) -> models.SparseVector:
    tokens = tokenize(text)
    if not tokens:
        return models.SparseVector(indices=[], values=[])
    tf = Counter(tokens)
    dl = len(tokens)
    # Collisions: two terms hashing to the same id -> sum their weights.
    weights: dict[int, float] = {}
    for term, freq in tf.items():
        w = freq * (K1 + 1) / (freq + K1 * (1 - B + B * dl / avgdl))
        tid = term_id(term)
        weights[tid] = weights.get(tid, 0.0) + w
    return models.SparseVector(indices=list(weights), values=list(weights.values()))


def query_sparse_vector(text: str) -> models.SparseVector:
    ids = sorted({term_id(t) for t in tokenize(text)})
    return models.SparseVector(indices=ids, values=[1.0] * len(ids))


def average_length(texts: list[str]) -> float:
    lengths = [len(tokenize(t)) for t in texts]
    return sum(lengths) / len(lengths) if lengths else 1.0
