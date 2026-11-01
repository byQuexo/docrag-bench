from pathlib import Path

import polars as pl
from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "mmlongbench"


def count_pdfs(directory: Path) -> list[str]:
    return [str(p) for p in (directory / "documents").glob("*.pdf")]


def main() -> None:
    root = snapshot_download(
        "yubo2333/MMLongBench-Doc",
        repo_type="dataset",
        local_dir=DATA_DIR,
        ignore_patterns="*mi_phone.pdf",
    )
    qa = pl.read_parquet(Path(root) / "data" / "*.parquet")

    pdfs = count_pdfs(DATA_DIR)

    pdf_ids = {Path(p).stem for p in pdfs}
    doc_ids = {d.removesuffix(".pdf") for d in qa["doc_id"].unique().to_list()}

    missing_on_disk = sorted(doc_ids - pdf_ids)

    print(
        f"Questions: {qa.height} | Documents in QA: {len(doc_ids)} "
        f"| PDFs on disk: {len(pdfs)}Missing on disk: {missing_on_disk}\n"
    )


if __name__ == "__main__":
    main()
