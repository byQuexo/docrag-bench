"""Generate fixtures/tiny.pdf and fixtures/tiny_scanned.pdf.

tiny.pdf: a 3-page report with headings, text and a ruled table.
tiny_scanned.pdf: the same pages as PICTURES only (no text layer), like a scan or
a slide deck exported as images - the input that needs OCR.

Generated (not downloaded) so the fixture is tiny, license-free and reproducible.
Run: uv run python scripts/make_tiny_pdf.py
"""

from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "fixtures" / "tiny.pdf"
OUT_SCANNED = ROOT / "fixtures" / "tiny_scanned.pdf"

INTRO = (
    "This report summarizes the Northwind customer survey. The survey was "
    "conducted by Acme Research between March and May 2024 among 1,200 customers "
    "in three regions. Respondents answered questions about product quality, "
    "delivery times and support."
)
RESULTS = (
    "Revenue grew in every region. Table 1 lists revenue by region in million "
    "euros. Europe remained the largest market, while Asia grew fastest."
)
CONCLUSION = (
    "The report recommends expanding the support team in Asia and shortening "
    "delivery times in Europe. Customer satisfaction should be measured again "
    "in 2025."
)
TABLE = [
    ["Region", "2023", "2024"],
    ["Europe", "41.0", "45.5"],
    ["Asia", "18.2", "24.9"],
    ["Americas", "30.1", "31.4"],
]


def heading(page: pymupdf.Page, y: float, text: str, size: float = 16) -> None:
    page.insert_text((72, y), text, fontsize=size, fontname="hebo")


def paragraph(page: pymupdf.Page, y: float, text: str) -> None:
    rect = pymupdf.Rect(72, y, 523, y + 120)
    page.insert_textbox(rect, text, fontsize=11, fontname="helv")


def table(page: pymupdf.Page, x: float, y: float) -> None:
    col_w, row_h = 120, 22
    for r, row in enumerate(TABLE):
        for c, cell in enumerate(row):
            cell_rect = pymupdf.Rect(
                x + c * col_w, y + r * row_h, x + (c + 1) * col_w, y + (r + 1) * row_h
            )
            # Drawn borders are what PyMuPDF's find_tables() detects.
            page.draw_rect(cell_rect, color=(0, 0, 0), width=0.8)
            page.insert_text(
                (cell_rect.x0 + 6, cell_rect.y1 - 7),
                cell,
                fontsize=10,
                fontname="hebo" if r == 0 else "helv",
            )


def main() -> None:
    doc = pymupdf.open()
    p1 = doc.new_page()  # A4-ish default (595 x 842)
    heading(p1, 90, "Northwind Survey Report", size=22)
    heading(p1, 140, "1 Introduction")
    paragraph(p1, 155, INTRO)

    p2 = doc.new_page()
    heading(p2, 90, "2 Results")
    paragraph(p2, 105, RESULTS)
    p2.insert_text((72, 190), "Table 1: Revenue by region (EUR million)", fontsize=10)
    table(p2, 72, 200)

    p3 = doc.new_page()
    heading(p3, 90, "3 Conclusion")
    paragraph(p3, 105, CONCLUSION)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc.save(OUT)
    print(f"wrote {OUT}")

    # Image-only copy: render each page to PNG and place it as a full-page image.
    scanned = pymupdf.open()
    for page in doc:
        png = page.get_pixmap(matrix=pymupdf.Matrix(2, 2)).tobytes("png")
        new = scanned.new_page(width=page.rect.width, height=page.rect.height)
        new.insert_image(new.rect, stream=png)
    scanned.save(OUT_SCANNED, deflate=True)
    print(f"wrote {OUT_SCANNED}")


if __name__ == "__main__":
    main()
