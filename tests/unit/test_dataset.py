"""MMLongBench loading: list parsing, slices, split by document, selection."""

from pathlib import Path

import polars as pl
import pytest

from docrag.eval.dataset import doc_split, load_questions, parse_list, select
from docrag.shared.config import DatasetConfig


def test_parse_list() -> None:
    assert parse_list("[19, 20]") == [19, 20]
    assert parse_list("['Table', 'Chart']") == ["Table", "Chart"]
    assert parse_list("[]") == [] and parse_list("") == []


@pytest.mark.parametrize("evil", ["__import__('os').getcwd()", "{'a': 1}"])
def test_parse_list_rejects_code_and_non_lists(evil: str) -> None:
    # literal_eval refuses function calls (ValueError) - this is WHY we don't
    # use eval(). A dict literal parses but isn't a list.
    with pytest.raises(ValueError):
        parse_list(evil)


def test_doc_split_is_deterministic_and_balanced() -> None:
    docs = [f"doc{i}" for i in range(1000)]
    splits = [doc_split(d) for d in docs]
    assert splits == [doc_split(d) for d in docs]  # same answer every run
    dev_share = splits.count("dev") / len(splits)
    assert 0.25 < dev_share < 0.35  # ~30% dev


@pytest.fixture
def qa_dir(tmp_path: Path) -> Path:
    """A tiny fake MMLongBench: a parquet file + PDFs (empty files suffice)."""
    rows = {
        "doc_id": ["a.pdf", "a.pdf", "b.pdf", "c.pdf"],
        "doc_type": ["r"] * 4,
        "question": ["q1 ", "q2", "q3", "q4"],
        "answer": ["x", "Not answerable", "3", "y"],
        "evidence_pages": ["[2]", "[]", "[4, 5]", "[1]"],
        "evidence_sources": [
            "['Table']",
            "[]",
            "['Chart', 'Pure-text (Plain-text)']",
            "['Generalized-text (Layout)']",
        ],
        "answer_format": ["Str", "None", "Int", "Str"],
    }
    pl.DataFrame(rows).write_parquet(tmp_path / "train.parquet")
    docs = tmp_path / "documents"
    docs.mkdir()
    for name in ("a", "b"):  # c.pdf is missing on disk on purpose
        (docs / f"{name}.pdf").write_bytes(b"")
    return tmp_path


def test_load_questions(qa_dir: Path) -> None:
    qs = load_questions(str(qa_dir / "*.parquet"))
    assert [q.doc_id for q in qs] == ["a", "a", "b", "c"]
    assert qs[0].question == "q1"  # stripped
    assert qs[0].slices == ["table"]
    assert qs[1].slices == ["unanswerable"] and not qs[1].answerable
    assert qs[2].slices == ["image", "text"] and qs[2].evidence_pages == [4, 5]
    assert qs[3].slices == ["text"]
    assert len({q.qid for q in qs}) == 4


def test_select_skips_missing_pdfs_and_applies_limits(qa_dir: Path) -> None:
    qs = load_questions(str(qa_dir / "*.parquet"))
    docs = qa_dir / "documents"
    chosen = select(qs, DatasetConfig(split="all", max_docs=None), docs)
    assert {q.doc_id for q in chosen} == {"a", "b"}
    assert len(select(qs, DatasetConfig(split="all", max_docs=1), docs)) == 2  # doc a
    limited = DatasetConfig(split="all", max_docs=None, max_questions=1)
    assert len(select(qs, limited, docs)) == 1
    # A split only ever contains documents of that split.
    for split in ("dev", "test"):
        part = select(qs, DatasetConfig(split=split, max_docs=None), docs)
        assert all(doc_split(q.doc_id) == split for q in part)
