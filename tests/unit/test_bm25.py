"""BM25 sparse vectors (document side + query side)."""

import pytest

from docrag.store.bm25 import doc_sparse_vector, query_sparse_vector, term_id


def _weights(text: str, avgdl: float = 10.0) -> dict[int, float]:
    v = doc_sparse_vector(text, avgdl)
    return dict(zip(v.indices, v.values, strict=True))


def test_term_frequency_saturates() -> None:
    once = _weights("revenue")[term_id("revenue")]
    many = _weights("revenue " * 10)[term_id("revenue")]
    # More occurrences -> higher weight, but far less than 10x (k1 saturation).
    assert once < many < 10 * once


def test_longer_documents_get_lower_weight() -> None:
    short = _weights("revenue europe")[term_id("revenue")]
    long = _weights("revenue " + "filler " * 30)[term_id("revenue")]
    assert long < short


def test_stopwords_and_empty() -> None:
    assert doc_sparse_vector("the of and", 10.0).indices == []
    assert query_sparse_vector("").indices == []


def test_query_vector_unique_terms_weight_one() -> None:
    q = query_sparse_vector("Revenue revenue Europe")
    assert sorted(q.indices) == sorted({term_id("revenue"), term_id("europe")})
    assert q.values == pytest.approx([1.0, 1.0])
