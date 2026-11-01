"""Load MMLongBench-Doc questions and tag them with evidence-type slices.

Dataset facts that shape this code:
* 1091 questions over 135 PDFs; one question is about exactly ONE document.
* ``evidence_pages`` are 1-based page numbers, stored as a *string* "[19, 20]".
* ~22% are "Not answerable" (empty evidence) - correct behaviour is to abstain.
* ``evidence_sources`` says where the answer is: text, layout, table, chart, figure.
"""

import ast
import hashlib
from pathlib import Path
from typing import Literal

import polars as pl
from pydantic import BaseModel

from docrag.answer.schema import NOT_ANSWERABLE
from docrag.shared.config import DatasetConfig

Slice = Literal["text", "table", "image", "unanswerable"]

_SOURCE_TO_SLICE: dict[str, Slice] = {
    "Pure-text (Plain-text)": "text",
    "Generalized-text (Layout)": "text",
    "Table": "table",
    "Chart": "image",
    "Figure": "image",
}


class EvalQuestion(BaseModel):
    qid: str
    doc_id: str
    question: str
    answer: str
    answer_format: str
    evidence_pages: list[int]
    slices: list[Slice]

    @property
    def answerable(self) -> bool:
        return self.answer.strip() != NOT_ANSWERABLE


def parse_list(raw: str) -> list[str | int]:
    """Parse "[19, 20]" or "['Table']" safely.

    ast.literal_eval only accepts Python LITERALS (numbers, strings, lists...).
    Never use eval() on data: eval("__import__('os').system('...')") would run it.
    """
    value = ast.literal_eval(raw) if raw and raw.strip() else []
    if not isinstance(value, list):
        raise ValueError(f"expected a list, got {raw!r}")
    return value


def doc_split(doc_id: str, dev_share: float = 0.3) -> Literal["dev", "test"]:
    """Deterministic split BY DOCUMENT (not by question).

    Splitting by question would put questions about the same PDF in dev and
    test; tuning the threshold on dev would then leak knowledge of test docs.
    """
    bucket = int(hashlib.sha256(doc_id.encode()).hexdigest(), 16) % 1000
    return "dev" if bucket < dev_share * 1000 else "test"


def load_questions(qa_glob: str) -> list[EvalQuestion]:
    df = pl.read_parquet(qa_glob)
    out: list[EvalQuestion] = []
    # LEARN: iter_rows(named=True) yields one dict per row. Fine for ~1k rows;
    # for millions you'd stay in polars' vectorized expressions instead.
    for row in df.iter_rows(named=True):
        doc_id = row["doc_id"].removesuffix(".pdf")
        sources = [str(s) for s in parse_list(row["evidence_sources"])]
        slices: list[Slice] = sorted(
            {_SOURCE_TO_SLICE[s] for s in sources if s in _SOURCE_TO_SLICE}
        )
        if row["answer"].strip() == NOT_ANSWERABLE:
            slices = ["unanswerable"]
        qid = hashlib.sha1(f"{doc_id}\n{row['question']}".encode()).hexdigest()[:12]
        out.append(
            EvalQuestion(
                qid=qid,
                doc_id=doc_id,
                question=row["question"].strip(),
                answer=str(row["answer"]).strip(),
                answer_format=row["answer_format"],
                evidence_pages=[int(p) for p in parse_list(row["evidence_pages"])],
                slices=slices,
            )
        )
    return out


def select(
    questions: list[EvalQuestion], cfg: DatasetConfig, pdf_dir: Path
) -> list[EvalQuestion]:
    """Apply split + limits. Only documents whose PDF exists on disk are used."""
    available = {p.stem for p in pdf_dir.glob("*.pdf")}
    docs = sorted(
        {
            q.doc_id
            for q in questions
            if q.doc_id in available
            and (cfg.split == "all" or doc_split(q.doc_id) == cfg.split)
        }
    )
    if cfg.max_docs is not None:
        docs = docs[: cfg.max_docs]
    keep = set(docs)
    chosen = [q for q in questions if q.doc_id in keep]
    if cfg.max_questions is not None:
        chosen = chosen[: cfg.max_questions]
    return chosen
