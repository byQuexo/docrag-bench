"""Threshold calibration and rerank filtering."""

import pytest

from docrag.rerank.reranker import FakeReranker, calibrate_threshold, rerank
from docrag.shared.schema import Chunk, Hit


def _hit(cid: str, text: str) -> Hit:
    chunk = Chunk(
        id=cid,
        doc_id="d",
        page_start=1,
        page_end=1,
        section_path=[],
        type="text",
        text=text,
        embed_text=text,
    )
    return Hit(chunk=chunk, score=0.0, source="hybrid")


def test_calibrate_perfect_separation() -> None:
    threshold, f1 = calibrate_threshold(
        [0.9, 0.8, 0.3, 0.1], [True, True, False, False]
    )
    assert threshold == pytest.approx(0.8)
    assert f1 == pytest.approx(1.0)


def test_calibrate_hand_computed_with_ties() -> None:
    # Threshold 0.9 keeps {T}:        P=1,   R=1/3, F1=0.5
    # Threshold 0.5 keeps {T, F, T}:  P=2/3, R=2/3, F1=2/3   (tie at 0.5!)
    # Threshold 0.2 keeps all four:   P=3/4, R=1,   F1=6/7  <- best
    threshold, f1 = calibrate_threshold([0.9, 0.5, 0.5, 0.2], [True, False, True, True])
    assert threshold == pytest.approx(0.2)
    assert f1 == pytest.approx(6 / 7)


def test_calibrate_tie_is_never_split() -> None:
    # With "score >= t" you can't keep one of two equal scores and drop the other.
    threshold, f1 = calibrate_threshold([0.7, 0.7, 0.1], [True, False, False])
    assert threshold == pytest.approx(0.7)
    assert f1 == pytest.approx(2 / 3)  # P=1/2, R=1


def test_calibrate_degenerate_inputs() -> None:
    assert calibrate_threshold([], []) == (0.5, 0.0)
    assert calibrate_threshold([0.3, 0.2], [False, False])[1] == 0.0


def test_rerank_filters_by_threshold_and_sorts() -> None:
    hits = [
        _hit("d:p1:c0", "unrelated text"),
        _hit("d:p1:c1", "revenue only"),
        _hit("d:p1:c2", "revenue europe"),
    ]
    out = rerank(FakeReranker(), "revenue europe", hits, top_k=5, threshold=0.5)
    assert [h.chunk.id for h in out] == ["d:p1:c2", "d:p1:c1"]
    assert [h.score for h in out] == [1.0, 0.5]
    assert all(h.source == "rerank" for h in out)


def test_rerank_top_k_and_empty() -> None:
    hits = [_hit(f"d:p1:c{i}", "revenue") for i in range(5)]
    assert len(rerank(FakeReranker(), "revenue", hits, top_k=2, threshold=0.0)) == 2
    assert rerank(FakeReranker(), "revenue", [], top_k=2, threshold=0.0) == []
