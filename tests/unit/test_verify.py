"""Verify: (a) cited ID in context, (b) quote verbatim in the chunk."""

from docrag.answer.schema import Answer, Evidence
from docrag.answer.verify import verify
from docrag.shared.schema import Chunk


def _chunk(cid: str, text: str, kind: str = "text") -> Chunk:
    return Chunk(
        id=cid,
        doc_id=cid.split(":")[0],
        page_start=int(cid.split(":p")[1].split(":")[0]),
        page_end=int(cid.split(":p")[1].split(":")[0]),
        section_path=[],
        type=kind,  # type: ignore[arg-type]
        text=text,
        embed_text=text,
    )


CONTEXT = [
    _chunk("tiny:p1:c0", "The survey was conducted by Acme Research in 2024."),
    _chunk("tiny:p2:c1", "Table 1\n|Region|2024|\n|---|---|\n|Europe|45.5|", "table"),
]


def _answer(text: str, *evidence: tuple[str, str]) -> Answer:
    return Answer(answer=text, evidence=[Evidence(id=i, quote=q) for i, q in evidence])


def test_supported() -> None:
    result = verify(
        "q",
        _answer(
            "Acme Research [tiny:p1:c0]", ("tiny:p1:c0", "conducted by Acme Research")
        ),
        CONTEXT,
    )
    assert result.status == "supported"
    assert result.citations[0].verified
    assert (result.citations[0].doc_id, result.citations[0].page) == ("tiny", 1)
    assert result.answer == "Acme Research [tiny:p1:c0]"


def test_id_not_in_context_is_rejected_and_stripped() -> None:
    result = verify(
        "q",
        _answer("Acme [tiny:p9:c0]", ("tiny:p9:c0", "conducted by Acme Research")),
        CONTEXT,
    )
    assert result.status == "unsupported"
    assert result.citations[0].reason == "id not in context"
    assert result.answer == "Acme"  # inline marker of the rejected citation removed
    assert result.raw_answer == "Acme [tiny:p9:c0]"


def test_paraphrased_quote_is_rejected() -> None:
    result = verify(
        "q",
        _answer("Acme [tiny:p1:c0]", ("tiny:p1:c0", "Acme Research ran the survey")),
        CONTEXT,
    )
    assert result.citations[0].reason == "quote not found in chunk"
    assert result.status == "unsupported"


def test_quote_normalization_quotes_case_and_markdown_table() -> None:
    result = verify(
        "q",
        _answer(
            "45.5 [tiny:p2:c1] by Acme [tiny:p1:c0]",
            ("tiny:p2:c1", "Europe 45.5"),  # cell boundaries "|" ignored
            ("tiny:p1:c0", "CONDUCTED  BY acme research"),  # case + spaces ignored
        ),
        CONTEXT,
    )
    assert [c.verified for c in result.citations] == [True, True]
    assert result.status == "supported"


def test_partially_supported() -> None:
    result = verify(
        "q",
        _answer(
            "A [tiny:p1:c0] B [tiny:p2:c1]",
            ("tiny:p1:c0", "Acme Research"),
            ("tiny:p2:c1", "Asia 99"),
        ),
        CONTEXT,
    )
    assert result.status == "partially_supported"
    assert result.answer == "A [tiny:p1:c0] B"
    assert [c.id for c in result.verified_citations] == ["tiny:p1:c0"]


def test_empty_quote_and_malformed_id() -> None:
    result = verify(
        "q",
        _answer("x", ("tiny:p1:c0", "   "), ("not-an-id", "Acme")),
        CONTEXT,
    )
    assert [c.reason for c in result.citations] == ["empty quote", "id not in context"]
    assert result.citations[1].page == 0


def test_abstained() -> None:
    result = verify("q", _answer("Not answerable."), CONTEXT)
    assert result.status == "abstained"
    assert result.context_ids == ["tiny:p1:c0", "tiny:p2:c1"]
