"""Bootstrap CI over documents and exact McNemar, against hand calculations."""

import math
from pathlib import Path

import pytest

from docrag.eval.stats import doc_bootstrap_ci, mcnemar


def test_mcnemar_hand_calculation() -> None:
    a = [True, True, False, False, True, False]
    b = [True, False, True, True, True, True]
    # b wins on 3 questions, loses on 1: p = 2 * P(X <= 1 | n=4, 0.5) = 2 * 5/16
    assert mcnemar(a, b) == (3, 1, pytest.approx(0.625))
    assert mcnemar(a, a) == (0, 0, 1.0)


def test_mcnemar_matches_a_known_result() -> None:
    # 19 wins vs 6 losses (the dev abstention experiment): p = 0.0146
    a = [False] * 19 + [True] * 6
    b = [True] * 19 + [False] * 6
    assert mcnemar(a, b)[2] == pytest.approx(0.01463, abs=1e-4)


def test_ci_contains_mean_and_is_reproducible() -> None:
    values = [1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0]
    docs = ["a", "a", "b", "b", "c", "c", "d", "d"]
    mean, lo, hi = doc_bootstrap_ci(values, docs)
    assert mean == pytest.approx(5 / 8)
    assert lo <= mean <= hi
    assert doc_bootstrap_ci(values, docs) == (mean, lo, hi)  # seed 0 -> same CI


def test_nan_is_dropped_and_one_doc_has_no_spread() -> None:
    mean, lo, hi = doc_bootstrap_ci([1.0, math.nan, 0.0], ["a", "a", "a"])
    assert (mean, lo, hi) == (0.5, 0.5, 0.5)
    assert all(math.isnan(x) for x in doc_bootstrap_ci([math.nan], ["a"]))


def test_clustered_data_gives_a_wider_interval_than_independent_data() -> None:
    # Same overall mean (0.5), but here each document is all-right or all-wrong.
    clustered = doc_bootstrap_ci([1.0] * 10 + [0.0] * 10, ["a"] * 10 + ["b"] * 10)
    spread = doc_bootstrap_ci([1.0, 0.0] * 10, [f"d{i}" for i in range(20)])
    assert (clustered[2] - clustered[1]) > (spread[2] - spread[1])


def test_summary_shows_bootstrap_cis(tmp_path: Path) -> None:
    import polars as pl

    from docrag.eval.runner import summarize

    run = tmp_path / "20260101T000000_demo_abc123"
    run.mkdir()
    pl.DataFrame(
        {
            "slice": ["all"],
            "n": [4],
            "recall@10": [0.5],
            "correct": [0.5],
            "run": ["demo"],
            "config_hash": ["abc123"],
        }
    ).write_csv(run / "metrics.csv")
    pl.DataFrame(
        {
            "qid": list("abcd"),
            "doc_id": ["x", "x", "y", "y"],
            "recall@10": ["1.0", "0.0", "1.0", "NaN"],
            "correct": ["true", "false", "true", ""],
        }
    ).write_csv(run / "per_question.csv")
    table = summarize(tmp_path)
    assert "recall@10 95% CI" in table and "correct 95% CI" in table
    assert "abc123" in table and "[" in table
