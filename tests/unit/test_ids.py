import uuid

import pytest

from docrag.shared.ids import chunk_point_id, make_chunk_id, parse_chunk_id


@pytest.mark.parametrize(
    ("doc_id", "page", "index", "expected"),
    [
        pytest.param("doc3", 2, 1, "doc3:p2:c1", id="simple"),
        pytest.param(
            "PH_2016.06.08_Economy-Final",
            5,
            0,
            "PH_2016.06.08_Economy-Final:p5:c0",
            id="real-doc-id",
        ),
        pytest.param("doc3", 1, 0, "doc3:p1:c0", id="lowest-valid-page-and-index"),
        pytest.param("doc3", 999, 42, "doc3:p999:c42", id="multi-digit"),
    ],
)
def test_make_chunk_id_valid(doc_id: str, page: int, index: int, expected: str) -> None:
    assert make_chunk_id(doc_id, page, index) == expected


@pytest.mark.parametrize(
    ("doc_id", "page", "index"),
    [
        pytest.param("", 2, 1, id="empty-doc-id"),
        pytest.param("doc3", 0, 1, id="page-zero"),
        pytest.param("doc:3", 2, 1, id="colon-in-doc-id"),
        pytest.param("doc 3", 2, 1, id="space-in-doc-id"),
        pytest.param("doc3", -1, 1, id="negative-page"),
        pytest.param("doc3", 2, -1, id="negative-index"),
        pytest.param("döc3", 2, 1, id="non-ascii"),
        pytest.param("[doc3]", 2, 1, id="brackets-in-doc-id"),
        pytest.param("doc3\n", 2, 1, id="trailing-newline"),
    ],
)
def test_make_chunk_id_rejects_invalid(doc_id: str, page: int, index: int) -> None:
    with pytest.raises(ValueError):
        make_chunk_id(doc_id, page, index)


def test_parse_chunk_id_roundtrip() -> None:
    cid = make_chunk_id("PH_2016.06.08_Economy-Final", 5, 3)
    assert parse_chunk_id(cid) == ("PH_2016.06.08_Economy-Final", 5, 3)


@pytest.mark.parametrize("bad", ["doc3", "doc3:p2", "doc3:pX:c1", "[doc3:p2:c1]", ""])
def test_parse_chunk_id_rejects_malformed(bad: str) -> None:
    with pytest.raises(ValueError):
        parse_chunk_id(bad)


def test_chunk_point_id_is_deterministic_uuid() -> None:
    # Same chunk id -> same Qdrant point id, so re-ingesting overwrites (no dupes).
    a, b = chunk_point_id("doc3:p2:c1"), chunk_point_id("doc3:p2:c1")
    assert a == b
    assert uuid.UUID(a).version == 5
    assert chunk_point_id("doc3:p2:c2") != a
