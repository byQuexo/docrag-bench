"""Retrieval and citation metrics at PAGE level.

The ground truth is per page ("evidence on pages 19 and 20"), but we retrieve
chunks. So: map the ranked chunks to pages, deduplicate (keep first occurrence),
then measure on the page ranking. Different chunkers stay comparable this way.
"""

import math

from docrag.answer.schema import Citation
from docrag.shared.schema import Hit


def ranked_pages(hits: list[Hit], doc_id: str | None = None) -> list[int]:
    """Chunks -> unique pages in rank order. A chunk on pages 4-5 yields 4, then 5.

    With ``doc_id`` (needed when search spans several documents, scope "corpus"),
    pages of OTHER documents keep their rank position but get a placeholder that
    can never match a gold page: page 5 of another PDF is not page 5 of this one.
    """
    seen: dict[tuple[str, int], None] = {}
    for hit in hits:
        for page in hit.chunk.pages:
            # LEARN: a dict keeps insertion order (guaranteed since 3.7), so
            # dict.fromkeys / setdefault is the idiomatic "ordered set".
            seen.setdefault((hit.chunk.doc_id, page), None)
    pages: list[int] = []
    for i, (doc, page) in enumerate(seen):
        pages.append(page if doc_id is None or doc == doc_id else -(i + 1))
    return pages


def recall_at_k(pages: list[int], gold: set[int], k: int) -> float:
    """Share of gold pages that appear in the top-k retrieved pages."""
    if not gold:
        return float("nan")
    return len(set(pages[:k]) & gold) / len(gold)


def mrr_at_k(pages: list[int], gold: set[int], k: int = 10) -> float:
    """1 / rank of the FIRST relevant page (0 if none in the top k)."""
    for rank, page in enumerate(pages[:k], start=1):
        if page in gold:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(pages: list[int], gold: set[int], k: int = 10) -> float:
    """nDCG@k with binary relevance (1 = gold page, 0 = not).

    DCG  = sum over retrieved ranks i (1-based) of rel_i / log2(i + 1)
    IDCG = the DCG of a perfect ranking: all gold pages first.
    nDCG = DCG / IDCG, in [0, 1].

    Subtle part: IDCG must count ALL gold pages (up to k), including ones we
    never retrieved. Otherwise retrieving 1 of 3 gold pages at rank 1 would
    score a perfect 1.0.

    Why not sklearn.metrics.ndcg_score (the plan's suggestion)? It only sees the
    items you pass in and ranks them by score. A first version appended the
    unretrieved gold pages with a low score - but when fewer than k pages were
    retrieved, those pages landed INSIDE sklearn's top-k and were counted as
    hits (caught by a unit test). The formula is four lines; we keep sklearn as
    a cross-check in tests/unit/test_metrics.py instead.
    """
    if not gold:
        return float("nan")
    dcg = sum(
        1.0 / math.log2(rank + 1) for rank, p in enumerate(pages[:k], 1) if p in gold
    )
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(gold), k) + 1))
    return dcg / idcg


def citation_precision(
    citations: list[Citation], gold: set[int], doc_id: str | None = None
) -> float:
    """Share of citations that point at a gold evidence page (NaN if none).

    With ``doc_id``, a citation must also be in that document (scope "corpus").
    """
    if not citations:
        return float("nan")
    hits = sum(
        1
        for c in citations
        if c.page in gold and (doc_id is None or c.doc_id == doc_id)
    )
    return hits / len(citations)
