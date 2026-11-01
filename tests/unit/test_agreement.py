"""Judge vs. human: the labeling sample is blind, kappa matches a hand calculation."""

import json
import math
from pathlib import Path

import polars as pl
import pytest

from docrag.eval.agreement import LABEL_COLUMNS, judge_agreement, label_sample
from docrag.eval.dataset import EvalQuestion
from docrag.store.qdrant import QdrantStore


def _run_dir(tmp_path: Path, collection: str) -> Path:
    run = tmp_path / "run"
    run.mkdir()
    pl.DataFrame(
        {
            "qid": ["a", "b", "c", "d", "e"],
            "answerable": ["true", "true", "true", "false", "true"],
            "abstained": ["false", "false", "false", "false", "true"],
            "answer": [
                "45.5 [tiny:p2:c1]",
                "Acme Research [tiny:p1:c0]",
                "Asia [tiny:p3:c0]",
                "x",
                "Not answerable",
            ],
            "correct": ["true", "false", "true", "false", "false"],
            "faithful": ["1.0", "1.0", "0.0", "NaN", "NaN"],
        }
    ).write_csv(run / "per_question.csv")
    (run / "run.json").write_text(json.dumps({"index": {"collection": collection}}))
    return run


def _questions() -> dict[str, EvalQuestion]:
    return {
        qid: EvalQuestion(
            qid=qid,
            doc_id="tiny",
            question=f"question {qid}",
            answer=f"gold {qid}",
            answer_format="Str",
            evidence_pages=[1],
            slices=["text"],
        )
        for qid in "abcde"
    }


def test_sample_is_blind_and_only_judged_answers(
    tmp_path: Path, memory_store: QdrantStore
) -> None:
    run = _run_dir(tmp_path, memory_store.collection)
    sample = label_sample(run, _questions(), memory_store.client, n=10)
    # d (unanswerable) and e (abstained) were decided by rules, not the judge.
    assert sorted(sample["qid"]) == ["a", "b", "c"]
    assert not {"correct", "faithful"} & set(sample.columns), "verdicts must be hidden"
    assert all(sample[c].to_list() == [""] * 3 for c in LABEL_COLUMNS)
    row = sample.filter(pl.col("qid") == "a").row(0, named=True)
    assert row["gold_answer"] == "gold a"
    assert "|Europe|41.0|45.5|" in row["cited_passages"], "full chunk text, not a quote"


def test_agreement_and_kappa(tmp_path: Path) -> None:
    run = _run_dir(tmp_path, "unused")
    labels = tmp_path / "labels.csv"
    pl.DataFrame(
        {
            "qid": ["a", "b", "c"],
            "human_correct": ["true", "true", "true"],  # disagrees on b
            "human_faithful": ["true", "true", ""],  # c not labeled yet
        }
    ).write_csv(labels, include_bom=True)
    report = judge_agreement(run, labels)
    assert report["correct"]["n"] == 3 and report["correct"]["agree"] == 2
    # Human always says True -> no variance on one side -> kappa = 0 (chance level).
    assert report["correct"]["kappa"] == pytest.approx(0.0)
    # Both raters said True for both labeled rows: kappa is undefined.
    assert (report["faithful"]["n"], report["faithful"]["agree"]) == (2, 2)
    assert math.isnan(report["faithful"]["kappa"])


def test_kappa_hand_calculation(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    judge = ["true", "true", "false", "false", "true", "false"]
    human = ["true", "false", "false", "false", "true", "true"]
    qids = [f"q{i}" for i in range(6)]
    pl.DataFrame({"qid": qids, "correct": judge, "faithful": ["NaN"] * 6}).write_csv(
        run / "per_question.csv"
    )
    labels = tmp_path / "labels.csv"
    pl.DataFrame(
        {"qid": qids, "human_correct": human, "human_faithful": [""] * 6}
    ).write_csv(labels)
    r = judge_agreement(run, labels)["correct"]
    # p_o = 4/6; both raters say True 3/6 -> p_e = 0.5; kappa = (4/6-0.5)/0.5 = 1/3
    assert (r["agree"], r["n"]) == (4, 6)
    assert r["kappa"] == pytest.approx(1 / 3, abs=1e-3)


def test_reads_what_german_excel_saves(tmp_path: Path) -> None:
    run = _run_dir(tmp_path, "unused")
    labels = tmp_path / "labels.csv"
    # ";" separator, Windows-1252 (umlauts), WAHR/FALSCH, a quoted multi-line cell.
    rows = [
        "qid;question;human_correct;human_faithful",
        'a;"Gr\u00f6\u00dfe\r\nzwei Zeilen";WAHR;WAHR',
        'b;"x; y";FALSCH;WAHR',
    ]
    labels.write_bytes("\r\n".join(rows).encode("cp1252") + b"\r\n")
    report = judge_agreement(run, labels)
    assert report["correct"]["n"] == 2 and report["correct"]["agree"] == 2
    assert report["faithful"]["n"] == 2


def test_interactive_labeling_saves_and_resumes(tmp_path: Path) -> None:
    path = tmp_path / "labels.csv"
    pl.DataFrame(
        {
            "qid": ["a", "b"],
            "question": ["q a", "q b"],
            "gold_answer": ["g", "g"],
            "system_answer": ["s", "s"],
            "cited_passages": ["p", "p"],
            "human_correct": ["", ""],
            "human_faithful": ["", ""],
        }
    ).write_csv(path)
    answers = iter(["maybe", "y", "n", "q"])  # invalid input is asked again
    from docrag.eval.agreement import label_interactive

    assert (
        label_interactive(path, ask=lambda _: next(answers), show=lambda _: None) == 2
    )
    saved = pl.read_csv(path, infer_schema_length=0)
    assert saved.row(0, named=True)["human_correct"] == "true"
    assert saved.row(0, named=True)["human_faithful"] == "false"
    # Resume: row a is done and skipped; row b gets y/y.
    rest = iter(["y", "y"])
    assert label_interactive(path, ask=lambda _: next(rest), show=lambda _: None) == 2
    assert pl.read_csv(path, infer_schema_length=0)["human_correct"].to_list() == [
        "true",
        "true",
    ]
