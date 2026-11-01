"""normalize() and tokenize(): the shared notion of "equal text"."""

import pytest

from docrag.shared.text import normalize, tokenize


# LEARN: @pytest.mark.parametrize runs the same test once per tuple - like JUnit's
# @ParameterizedTest or a table-driven test in Go/Rust. Each case shows up as its
# own test in the report, so one failing case doesn't hide the others.
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("The  \u201cRevenue\u201d", 'the "revenue"'),
        ("infor-\nmation", "information"),
        ("a\u00a0b", "a b"),
        ("\ufb01nance", "finance"),  # the "fi" ligature, common in PDFs
        ("2023\u20132024", "2023-2024"),
        ("  lots\n\tof   space ", "lots of space"),
    ],
)
def test_normalize(raw: str, expected: str) -> None:
    assert normalize(raw) == expected


def test_tokenize_drops_stopwords_and_keeps_numbers_whole() -> None:
    assert tokenize("The revenue of 1,500 was 3.3 in Europe") == [
        "revenue",
        "1,500",
        "3.3",
        "europe",
    ]


def test_tokenize_empty_and_stopword_only() -> None:
    assert tokenize("") == []
    assert tokenize("the of and") == []
