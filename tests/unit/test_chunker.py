"""chunk_blocks: table-aware, page-preserving, stable IDs."""

from pathlib import Path

from docrag.ingest.chunker import ChunkerOptions, chunk_blocks
from docrag.ingest.pipeline import ingest_pdf
from docrag.shared.schema import Block, Chunk


def _text(page: int, text: str, section: list[str], order: int) -> Block:
    return Block(
        doc_id="d", type="text", text=text, page=page, section_path=section, order=order
    )


def test_tiny_pdf_chunks(tiny_chunks: list[Chunk]) -> None:
    assert [c.id for c in tiny_chunks] == [
        "tiny:p1:c0",
        "tiny:p2:c0",
        "tiny:p2:c1",
        "tiny:p3:c0",
    ]
    assert all(c.page_start >= 1 and c.page_end >= c.page_start for c in tiny_chunks)


def test_table_stays_one_chunk_with_caption(tiny_chunks: list[Chunk]) -> None:
    tables = [c for c in tiny_chunks if c.type == "table"]
    assert len(tables) == 1
    table = tables[0]
    assert table.text.startswith("Table 1: Revenue by region")
    for row in ("Europe", "Asia", "Americas"):
        assert row in table.text
    assert table.pages == [2]


def test_prefix_only_changes_embed_text(tiny_pdf: Path) -> None:
    with_prefix = ingest_pdf(tiny_pdf, "tiny", prefix_headings=True)
    without = ingest_pdf(tiny_pdf, "tiny", prefix_headings=False)
    for a, b in zip(with_prefix, without, strict=True):
        assert a.id == b.id and a.text == b.text  # raw text unchanged
        assert a.embed_text.startswith("Northwind Survey Report > ")
        assert b.embed_text == b.text


def test_ids_stable_across_runs(tiny_pdf: Path) -> None:
    assert ingest_pdf(tiny_pdf, "tiny") == ingest_pdf(tiny_pdf, "tiny")


def test_section_change_starts_new_chunk() -> None:
    blocks = [
        _text(1, "alpha beta", ["A"], 0),
        _text(1, "gamma delta", ["B"], 1),
    ]
    chunks = chunk_blocks(blocks)
    assert [c.text for c in chunks] == ["alpha beta", "gamma delta"]
    assert [c.section_path for c in chunks] == [["A"], ["B"]]


def test_paragraph_across_pages_keeps_both_pages() -> None:
    blocks = [_text(4, "first half", ["S"], 0), _text(5, "second half", ["S"], 1)]
    (chunk,) = chunk_blocks(blocks)
    assert (chunk.page_start, chunk.page_end) == (4, 5)
    assert chunk.id == "d:p4:c0"


def test_long_text_is_split_under_max_words() -> None:
    sentence = "word " * 9 + "end."
    blocks = [_text(1, " ".join([sentence] * 30), ["S"], 0)]  # 300 words
    chunks = chunk_blocks(blocks, ChunkerOptions(max_words=50))
    assert len(chunks) >= 6
    assert all(len(c.text.split()) <= 50 for c in chunks)
    assert [c.id for c in chunks][:2] == ["d:p1:c0", "d:p1:c1"]


def test_big_table_split_repeats_caption_and_header() -> None:
    header = "|Name|Value|\n|---|---|"
    rows = "\n".join(f"|row{i}|{i}|" for i in range(60))
    md = f"{header}\n{rows}"
    block = Block(
        doc_id="d",
        type="table",
        text=f"Table 9: Big\n{md}",
        table_md=md,
        caption="Table 9: Big",
        page=3,
        section_path=[],
        order=0,
    )
    chunks = chunk_blocks([block], ChunkerOptions(max_table_words=20))
    assert len(chunks) > 1
    for c in chunks:
        lines = c.text.splitlines()
        assert lines[:3] == ["Table 9: Big", "|Name|Value|", "|---|---|"]
        assert c.type == "table" and c.pages == [3]
    # Every row survives exactly once across the pieces.
    all_rows = [ln for c in chunks for ln in c.text.splitlines()[3:]]
    assert all_rows == rows.splitlines()
