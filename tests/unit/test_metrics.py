"""Page-level retrieval metrics, checked against hand calculations."""

import math

import pytest

from docrag.answer.schema import Citation
from docrag.eval.metrics import (
    citation_precision,
    mrr_at_k,
    ndcg_at_k,
    ranked_pages,
    recall_at_k,
)
from docrag.shared.schema import Chunk, Hit


def _hit(page_start: int, page_end: int, idx: int = 0) -> Hit:
    chunk = Chunk(
        id=f"d:p{page_start}:c{idx}",
        doc_id="d",
        page_start=page_start,
        page_end=page_end,
        section_path=[],
        type="text",
        text="t",
        embed_text="t",
    )
    return Hit(chunk=chunk, score=1.0, source="dense")


def test_ranked_pages_dedupes_and_expands_spans() -> None:
    hits = [_hit(4, 5), _hit(4, 4, 1), _hit(2, 2), _hit(5, 6)]
    assert ranked_pages(hits) == [4, 5, 2, 6]


def test_recall() -> None:
    pages = [3, 7, 1, 9]
    assert recall_at_k(pages, {7, 9}, 1) == 0.0
    assert recall_at_k(pages, {7, 9}, 2) == 0.5
    assert recall_at_k(pages, {7, 9}, 10) == 1.0
    assert math.isnan(recall_at_k(pages, set(), 5))  # unanswerable: not applicable


def test_mrr() -> None:
    assert mrr_at_k([3, 7, 1], {7, 1}) == 0.5  # first relevant at rank 2
    assert mrr_at_k([3, 7, 1], {1}, k=2) == 0.0  # outside the cutoff
    assert mrr_at_k([7], {7}) == 1.0


def test_ndcg_hand_computed() -> None:
    # Ranking [3, 7, 9], gold {7, 9}:
    #   DCG  = 1/log2(3) + 1/log2(4)        (ranks 2 and 3)
    #   IDCG = 1/log2(2) + 1/log2(3)        (ideal: both at ranks 1 and 2)
    dcg = 1 / math.log2(3) + 1 / math.log2(4)
    idcg = 1 / math.log2(2) + 1 / math.log2(3)
    assert ndcg_at_k([3, 7, 9], {7, 9}) == pytest.approx(dcg / idcg)


def test_ndcg_counts_unretrieved_gold_pages() -> None:
    # 1 of 3 gold pages found, at rank 1. A naive implementation scores 1.0;
    # correct: DCG = 1, IDCG = 1 + 1/log2(3) + 1/log2(4).
    idcg = 1 + 1 / math.log2(3) + 1 / math.log2(4)
    assert ndcg_at_k([5, 8], {5, 6, 7}) == pytest.approx(1 / idcg)


def test_ndcg_edge_cases() -> None:
    assert ndcg_at_k([1, 2, 3], {1}) == pytest.approx(1.0)
    assert ndcg_at_k([], {9}) == 0.0
    assert math.isnan(ndcg_at_k([1], set()))


def test_ndcg_nothing_relevant_retrieved_is_zero() -> None:
    assert ndcg_at_k([2, 3], {9}) == 0.0


def _cit(page: int) -> Citation:
    return Citation(id=f"d:p{page}:c0", quote="q", doc_id="d", page=page, verified=True)


def test_citation_precision() -> None:
    assert citation_precision([_cit(2), _cit(3), _cit(2)], {2}) == pytest.approx(2 / 3)
    assert math.isnan(citation_precision([], {2}))


def test_ndcg_matches_sklearn_when_all_gold_pages_were_ranked() -> None:
    """Cross-check against sklearn in the case where both definitions agree.

    sklearn only knows the items it is given; when every gold page appears in the
    ranked list (and the list has >= k items), its ideal DCG equals ours.
    """
    import numpy as np
    from sklearn.metrics import ndcg_score

    pages = [4, 1, 7, 2, 9, 3, 8, 5, 6, 10]
    gold = {1, 2, 3}
    y_true = np.asarray([[1.0 if p in gold else 0.0 for p in pages]])
    y_score = np.asarray([[float(len(pages) - i) for i in range(len(pages))]])
    assert ndcg_at_k(pages, gold, 10) == pytest.approx(
        ndcg_score(y_true, y_score, k=10)
    )


def test_pages_of_other_documents_never_count_as_gold() -> None:
    other = Hit(
        chunk=_hit(2, 2).chunk.model_copy(update={"doc_id": "other"}),
        score=1.0,
        source="dense",
    )
    hits = [other, _hit(5, 5)]
    # Without doc_id, page 2 of "other" would look like gold page 2.
    assert ranked_pages(hits) == [2, 5]
    pages = ranked_pages(hits, doc_id="d")
    assert pages[1] == 5 and pages[0] < 0, "other doc keeps its rank, never matches"
    assert recall_at_k(pages, {2}, 10) == 0.0
    assert mrr_at_k(pages, {5}) == pytest.approx(1 / 2), (
        "the other doc pushed it to rank 2"
    )
    cit = Citation(id="other:p2:c0", quote="q", doc_id="other", page=2, verified=True)
    assert citation_precision([cit], {2}) == 1.0
    assert citation_precision([cit], {2}, doc_id="d") == 0.0
