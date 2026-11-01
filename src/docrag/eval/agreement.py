"""How far can we trust the LLM judge? Compare its verdicts with human labels.

Two steps, with a human in between:

1. ``label_sample``: draw N answers the judge MODEL graded (answerable, not
   abstained) from one run and write a CSV to label by hand. It is BLIND: the
   judge's verdicts are not in it, so they can't anchor the human.
2. ``judge_agreement``: join the filled-in CSV with the run's verdicts and report
   raw agreement and Cohen's kappa for ``correct`` and ``faithful``.

Why kappa and not just "% agree"? If the judge says "correct" 80% of the time
and so does the human, they agree ~68% of the time by pure chance. Kappa is
agreement corrected for chance: 0 = chance level, 1 = perfect.
"""

import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import polars as pl
from qdrant_client import QdrantClient
from sklearn.metrics import cohen_kappa_score

from docrag.eval.dataset import EvalQuestion
from docrag.shared.ids import chunk_point_id
from docrag.shared.schema import Chunk

LABEL_COLUMNS = ["human_correct", "human_faithful"]
_CITED_ID = re.compile(r"\[([^\]\s]+:p\d+:c\d+)\]")
# German Excel turns typed true/false into WAHR/FALSCH when it saves.
_TRUE = {"true", "1.0", "1", "yes", "y", "wahr"}
_FALSE = {"false", "0.0", "0", "no", "n", "falsch"}


def _bool(value: Any) -> bool | None:
    text = str(value).strip().lower()
    return True if text in _TRUE else False if text in _FALSE else None


def judged_rows(per_question: pl.DataFrame) -> pl.DataFrame:
    """Rows where the judge model read the answer and returned a verdict."""
    return per_question.filter(
        (pl.col("answerable").cast(pl.Utf8).str.to_lowercase() == "true")
        & (pl.col("abstained").cast(pl.Utf8).str.to_lowercase() == "false")
        & pl.col("correct").is_not_null()
    )


def label_sample(
    run_dir: Path,
    questions: dict[str, EvalQuestion],
    client: QdrantClient,
    n: int = 50,
    seed: int = 0,
) -> pl.DataFrame:
    """N judged answers of ``run_dir``: everything a human needs, minus the verdicts."""
    rows = judged_rows(pl.read_csv(run_dir / "per_question.csv", infer_schema_length=0))
    sample = rows.sample(n=min(n, rows.height), seed=seed)
    collection = json.loads((run_dir / "run.json").read_text())["index"]["collection"]
    out = []
    for row in sample.iter_rows(named=True):
        q = questions[row["qid"]]
        # Same passages judge_v2 saw: the chunks behind the verified citations
        # (rejected citations were already stripped from the answer text).
        ids = list(dict.fromkeys(_CITED_ID.findall(row["answer"] or "")))
        points = client.retrieve(collection, [chunk_point_id(i) for i in ids])
        passages = [Chunk.model_validate(p.payload) for p in points]
        out.append(
            {
                "qid": q.qid,
                "question": q.question,
                "gold_answer": q.answer,
                "system_answer": row["answer"],
                "cited_passages": "\n\n".join(f"[{c.id}]\n{c.text}" for c in passages)
                or "(none)",
                **dict.fromkeys(LABEL_COLUMNS, ""),
            }
        )
    return pl.DataFrame(out)


def read_labels(path: Path) -> pl.DataFrame:
    """Read a label CSV the way spreadsheets actually save it.

    German Excel writes ``;`` as separator, and "CSV" (not "CSV UTF-8") is
    Windows-1252. Both are detected here instead of failing on the human's file.
    """
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("cp1252")
    header = text.split("\n", 1)[0]
    sep = ";" if header.count(";") > header.count(",") else ","
    return pl.read_csv(text.encode("utf-8"), separator=sep, infer_schema_length=0)


_QUESTIONS = {
    "human_correct": "CORRECT? Same meaning as the gold answer",
    "human_faithful": "FAITHFUL? Every claim is in (or follows from) the passages",
}


def label_interactive(
    path: Path,
    ask: Callable[[str], str] = input,
    show: Callable[[str], None] = print,
) -> int:
    """Label a sample in the terminal, one answer at a time.

    Saves after every answer, so you can quit (q) and resume later; rows that
    already have both labels are skipped. Returns how many labels were set.
    """
    df = read_labels(path)
    rows = df.to_dicts()
    todo = [r for r in rows if any(_bool(r[c]) is None for c in LABEL_COLUMNS)]
    done = len(rows) - len(todo)
    set_count = 0
    for i, row in enumerate(todo, start=done + 1):
        show(
            f"\n{'=' * 78}\n[{i}/{len(rows)}]  {row['qid']}\n\n"
            f"QUESTION: {row['question']}\n\nGOLD:     {row['gold_answer']}\n\n"
            f"SYSTEM:   {row['system_answer']}\n\n"
            f"CITED PASSAGES:\n{row['cited_passages']}\n{'-' * 78}"
        )
        for col in LABEL_COLUMNS:
            if _bool(row[col]) is not None:
                continue
            while True:
                answer = ask(f"{_QUESTIONS[col]}  [y/n, s=skip, q=quit]: ")
                answer = answer.strip().lower()
                if answer in {"y", "n", "s", "q"}:
                    break
            if answer == "q":
                return set_count
            if answer != "s":
                row[col] = "true" if answer == "y" else "false"
                set_count += 1
                pl.DataFrame(rows).write_csv(path, include_bom=True)
    return set_count


def judge_agreement(run_dir: Path, labels_csv: Path) -> dict[str, dict[str, float]]:
    """Agreement + Cohen's kappa per field, over the rows a human has labeled."""
    labels = read_labels(labels_csv)
    judge = pl.read_csv(run_dir / "per_question.csv", infer_schema_length=0)
    joined = labels.join(judge.select("qid", "correct", "faithful"), on="qid")
    report: dict[str, dict[str, float]] = {}
    for field in ("correct", "faithful"):
        pairs = [
            (j, h)
            for j, h in (
                (_bool(r[field]), _bool(r[f"human_{field}"]))
                for r in joined.iter_rows(named=True)
            )
            if j is not None and h is not None
        ]
        if not pairs:
            continue
        judge_v, human_v = zip(*pairs, strict=True)
        agree = sum(j == h for j, h in pairs)
        # Undefined when both raters only ever use the same single label.
        kappa = (
            float(cohen_kappa_score(human_v, judge_v, labels=[False, True]))
            if len(set(human_v) | set(judge_v)) == 2
            else float("nan")
        )
        report[field] = {"n": len(pairs), "agree": agree, "kappa": round(kappa, 3)}
    return report
