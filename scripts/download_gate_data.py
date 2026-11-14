"""Download only what the CI quality gate needs: the question table + 4 PDFs.

    uv run python scripts/download_gate_data.py data/ci_gate

Writes <out>/data/*.parquet and <out>/documents/<doc_id>.pdf, the same layout
as the full download (scripts/download_data.py), so the gate is pointed at it
with DOCRAG_QA_GLOB and DOCRAG_PDF_DIR.
"""

import argparse
from pathlib import Path
from typing import Any

import yaml
from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "fixtures" / "ci_gate.yaml"


def load_spec(path: Path = SPEC) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        spec: dict[str, Any] = yaml.safe_load(f)
    return spec


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    spec = load_spec()
    patterns = ["data/*.parquet"] + [
        f"documents/{doc}.pdf" for doc in spec["documents"]
    ]
    snapshot_download(
        spec["dataset"]["repo"],
        repo_type="dataset",
        revision=spec["dataset"]["revision"],
        local_dir=args.out,
        allow_patterns=patterns,
    )
    pdfs = sorted(p.stem for p in (args.out / "documents").glob("*.pdf"))
    missing = sorted(set(spec["documents"]) - set(pdfs))
    if missing:
        raise SystemExit(f"missing PDFs after download: {missing}")
    print(f"{len(pdfs)} PDFs + questions in {args.out}")


if __name__ == "__main__":
    main()
