"""PyMuPdfParser on fixtures/tiny.pdf: pages, headings, the table."""

from pathlib import Path

import pytest

from docrag.ingest.parser import get_parser
from docrag.shared.schema import Block


@pytest.fixture(scope="module")
def blocks(tiny_pdf: Path) -> list[Block]:
    return get_parser("pymupdf").parse(tiny_pdf.read_bytes(), "tiny")


def test_every_block_has_a_1_based_page(blocks: list[Block]) -> None:
    assert {b.page for b in blocks} == {1, 2, 3}


def test_heading_chain(blocks: list[Block]) -> None:
    by_page = {b.page: b.section_path for b in blocks if b.type == "text"}
    assert by_page[1] == ["Northwind Survey Report", "1 Introduction"]
    # "2 Results" replaces "1 Introduction" (same level), the title stays.
    assert by_page[2] == ["Northwind Survey Report", "2 Results"]
    assert by_page[3] == ["Northwind Survey Report", "3 Conclusion"]


def test_table_is_markdown_with_caption(blocks: list[Block]) -> None:
    tables = [b for b in blocks if b.type == "table"]
    assert len(tables) == 1
    table = tables[0]
    assert table.page == 2
    assert table.caption is not None and table.caption.startswith("Table 1")
    assert table.table_md is not None and "|Europe|41.0|45.5|" in table.table_md
    # Table cell text must not also appear as a separate text block.
    texts = " ".join(b.text for b in blocks if b.type == "text")
    assert "45.5" not in texts


def test_reading_order_is_increasing(blocks: list[Block]) -> None:
    orders = [b.order for b in blocks]
    assert orders == sorted(orders)
    assert len(set(orders)) == len(orders)


def test_unknown_parser_name() -> None:
    with pytest.raises(ValueError, match="Unknown parser"):
        get_parser("docling")
