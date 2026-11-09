"""Recompute every statistic the README reports, from results/ only.

    uv run python scripts/reproduce_stats.py            # print Markdown
    uv run python scripts/reproduce_stats.py --out x.md # also write it to a file

No API key, no network, no index: it reads the per_question.csv, run.json and
config.yaml of the newest run per config (the same selection `docrag summary`
and the dashboard use) and the human labels in results/human_labels/. Intervals
come from ``stats.doc_bootstrap_ci`` (documents resampled, 2,000 reps, seed 0),
paired tests from ``stats.mcnemar``. Paired runs are aligned by row: every test
run lists the same 735 questions in the same order (one qid appears twice).
"""

import argparse
import io
import json
import math
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import polars as pl

from docrag.eval.agreement import _bool, judge_agreement, read_labels
from docrag.eval.metrics import mrr_at_k, ndcg_at_k, recall_at_k
from docrag.eval.overview import run_dirs, run_name, values
from docrag.eval.stats import doc_bootstrap_ci, mcnemar

ROOT = Path(__file__).resolve().parents[1]

# README row label -> config name (the newest run of that name is used).
ANSWER_ROWS = {
    "hybrid + rerank + LLM": "answer_haiku",
    "+ OCR (RapidOCR)": "answer_haiku_ocr",
    "+ OCR (vision LLM)": "answer_haiku_vlm",
    "+ prompt answer_v2, 10 chunks": "answer_haiku_vlm_v2",
    "+ prompt answer_v3, 10 chunks": "answer_haiku_vlm_v3",
}
RETRIEVAL_ROWS = {
    "dense (gte-small)": "dense",
    "BM25": "bm25",
    "hybrid (RRF)": "hybrid",
    "hybrid + rerank": "baseline",
    "hybrid + rerank, no heading prefix": "no_prefix",
    "hybrid + rerank, gte-modernbert-base": "modernbert",
    "hybrid + rerank + RapidOCR": "baseline_ocr",
    "hybrid + rerank + vision-LLM OCR": "baseline_vlm",
}
MODEL_ROWS = {
    "Claude Haiku 5.5": "answer_haiku_vlm_v3",
    "Gemini 3.8 Flash": "answer_model_gemini38flash",
    "Qwen3.8-27B": "answer_model_qwen38_27b",
    "Qwen3.8 Flash": "answer_model_qwen38flash",
    "GPT-5.4 mini": "answer_model_gpt54mini",
    "Qwen3.5-9B": "answer_model_qwen35_9b",
    "Mistral Small 2603": "answer_model_mistralsmall",
}


def fmt(x: float | None, nd: int = 3) -> str:
    return "–" if x is None or math.isnan(x) else f"{x:.{nd}f}"


def mean(v: list[float]) -> float:
    v = [x for x in v if not math.isnan(x)]
    return sum(v) / len(v) if v else math.nan


def ci_str(df: pl.DataFrame, col: str) -> str:
    m, lo, hi = doc_bootstrap_ci(values(df, col), df["doc_id"].to_list())
    return f"{fmt(m)} [{fmt(lo)}, {fmt(hi)}]"


def p_str(p: float) -> str:
    return "p < 0.0001" if p < 1e-4 else "p < 0.001" if p < 1e-3 else f"p = {p:.3f}"


class Results:
    def __init__(self, results_dir: Path) -> None:
        self.dir = results_dir
        self.latest = {run_name(d): d for d in run_dirs(results_dir)}
        self.frames = {
            n: pl.read_csv(d / "per_question.csv", infer_schema_length=0)
            for n, d in self.latest.items()
        }
        # Same rule as the dashboard: a document is "scanned" if any run tagged it.
        self.scanned: set[str] = set()
        for df in self.frames.values():
            tagged = df.filter(pl.col("slices").str.contains("no_text_layer"))
            self.scanned |= set(tagged["doc_id"].to_list())

    def df(self, name: str) -> pl.DataFrame:
        return self.frames[name]

    def info(self, name: str) -> dict[str, Any]:
        path = self.latest[name] / "run.json"
        return dict(json.loads(path.read_text(encoding="utf-8")))

    def part(self, name: str, which: str) -> pl.DataFrame:
        df = self.df(name)
        tags = pl.col("slices").str.split(",")
        doc = pl.col("doc_id").is_in(list(self.scanned))
        ans = pl.col("answerable").str.to_lowercase()
        return {
            "all": df,
            "answerable": df.filter(ans == "true"),
            "unanswerable": df.filter(ans == "false"),
            "table": df.filter(tags.list.contains("table")),
            "image": df.filter(tags.list.contains("image")),
            "scanned": df.filter(doc),
            "text_layer": df.filter(~doc),
        }[which]

    def col(self, name: str, col: str, which: str = "all") -> list[float]:
        return values(self.part(name, which), col)


def provenance(r: Results) -> list[str]:
    out = ["## Provenance of the runs used (run.json → code)", ""]
    out += ["| run folder | commit | dirty |", "|---|---|---|"]
    for name, d in r.latest.items():
        code = r.info(name).get("code") or {}
        commit = (code.get("commit") or "")[:7] or "not recorded"
        out.append(f"| `{d.name}` | {commit} | {code.get('dirty', '–')} |")
    return out + [""]


def split(r: Results) -> list[str]:
    df = r.df("answer_haiku_vlm_v3")
    scanned = r.part("answer_haiku_vlm_v3", "scanned")
    unans = r.part("answer_haiku_vlm_v3", "unanswerable").height
    return [
        "## Test split",
        "",
        f"- {df.height} questions over {df['doc_id'].n_unique()} documents; "
        f"{unans} unanswerable ({unans / df.height:.1%})",
        f"- scanned (no text layer): {scanned['doc_id'].n_unique()} documents, "
        f"{scanned.height} questions",
        "",
    ]


def answers(r: Results) -> list[str]:
    out = ["## Answers (test split)", ""]
    out += [
        "| variant | correct [95% CI] | answerable | unanswerable | tables | charts "
        "| scanned PDFs | faithful | citation precision | judge_failed |",
        "|---" * 10 + "|",
    ]
    for label, name in ANSWER_ROWS.items():
        cells = [
            ci_str(r.df(name), "correct"),
            *(
                fmt(mean(values(r.part(name, w), "correct")))
                for w in ("answerable", "unanswerable", "table", "image", "scanned")
            ),
            fmt(mean(values(r.df(name), "faithful"))),
            fmt(mean(values(r.df(name), "cit_precision_verify_on"))),
            str(int(sum(x == 1.0 for x in values(r.df(name), "judge_failed")))),
        ]
        out.append(f"| {label} | " + " | ".join(cells) + " |")
    return out + [""]


def paired(r: Results, a: str, b: str, which: str = "all") -> str:
    # Rows where either judge failed (no verdict) are left out, as in the means.
    # A failed ANSWER is a verdict (correct = false) and stays in.
    pairs = [
        (x == 1.0, y == 1.0)
        for x, y in zip(
            r.col(a, "correct", which), r.col(b, "correct", which), strict=True
        )
        if not (math.isnan(x) or math.isnan(y))
    ]
    if r.df(a)["qid"].to_list() != r.df(b)["qid"].to_list():
        raise ValueError(f"{a} and {b} don't list the same questions in order")
    w, losses, p = mcnemar([x for x, _ in pairs], [y for _, y in pairs])
    return f"+{w}/−{losses}, McNemar {p_str(p)}"


def consistency(r: Results) -> list[str]:
    """Stored per-question retrieval metrics vs. today's metric code on the stored
    rankings (top_pages, gold_pages). 0 mismatches = no metric drift since the run."""
    out = ["## Consistency: stored retrieval metrics vs current metric code", ""]
    checks: dict[str, Callable[[list[int], set[int]], float]] = {
        "recall@1": lambda p, g: recall_at_k(p, g, 1),
        "recall@10": lambda p, g: recall_at_k(p, g, 10),
        "mrr@10": lambda p, g: mrr_at_k(p, g, 10) if g else math.nan,
        "ndcg@10": lambda p, g: ndcg_at_k(p, g, 10),
    }
    bad = total = 0
    for df in r.frames.values():
        for row in df.iter_rows(named=True):
            pages, gold = (
                json.loads(row["top_pages"]),
                set(json.loads(row["gold_pages"])),
            )
            for col, f in checks.items():
                stored, now = float(row[col]), f(pages, gold)
                total += 1
                same = (math.isnan(stored) and math.isnan(now)) or abs(
                    stored - now
                ) < 1e-9
                bad += not same
    out.append(f"- {bad} mismatches in {total} values over {len(r.frames)} runs")
    return out + [""]


def diff_ci(r: Results, a: str, b: str, col: str, which: str = "all") -> str:
    da, db = r.part(a, which), r.part(b, which)
    diffs = [y - x for x, y in zip(values(da, col), values(db, col), strict=True)]
    m, lo, hi = doc_bootstrap_ci(diffs, da["doc_id"].to_list())
    return f"{m:+.3f} [{lo:+.3f}, {hi:+.3f}]"


def abstention(r: Results) -> list[str]:
    vlm, v2, v3 = "answer_haiku_vlm", "answer_haiku_vlm_v2", "answer_haiku_vlm_v3"
    out = ["## Abstention experiment (paired, test split)", ""]
    for label, a, b in [("v2 vs vlm", vlm, v2), ("v3 vs v2", v2, v3)]:
        for which in ("all", "answerable", "unanswerable"):
            out.append(f"- {label}, {which}: {paired(r, a, b, which)}")
    out.append(
        f"- v3 vs vlm, all: {paired(r, vlm, v3)}; "
        f"Δ correct {diff_ci(r, vlm, v3, 'correct')}"
    )
    gate = mean(values(r.part(v3, "answerable"), "abstained_before_llm"))
    out.append(f"- v3: answerable questions stopped by the rerank gate: {gate:.3f}")
    out += ["", "Dev split (where the variants were compared):", ""]
    d1, d2, d3 = (
        "dev_vlm_answer_v1_k5",
        "dev_vlm_answer_v2_k10",
        "dev_vlm_answer_v3_k10",
    )
    for name in (d1, d2, d3):
        cells = ", ".join(
            f"{w} {fmt(mean(values(r.part(name, w), 'correct')))}"
            for w in ("all", "answerable", "unanswerable")
        )
        out.append(f"- {name}: correct {cells}")
    for label, a, b in [("v2_k10 vs v1_k5", d1, d2), ("v3_k10 vs v2_k10", d2, d3)]:
        for which in ("all", "answerable", "unanswerable"):
            out.append(f"- dev {label}, {which}: {paired(r, a, b, which)}")
    return out + [""]


def retrieval(r: Results) -> list[str]:
    out = ["## Retrieval (text-layer answerable questions; scanned separately)", ""]
    out += [
        "| variant | n text | recall@1 | recall@10 [95% CI] | MRR@10 "
        "| n scanned | recall@10 scanned [95% CI] |",
        "|---" * 7 + "|",
    ]
    for label, name in RETRIEVAL_ROWS.items():
        t = r.part(name, "text_layer")
        s = r.part(name, "scanned")
        n_t = sum(not math.isnan(x) for x in values(t, "recall@10"))
        n_s = sum(not math.isnan(x) for x in values(s, "recall@10"))
        out.append(
            f"| {label} | {n_t} | {fmt(mean(values(t, 'recall@1')))} "
            f"| {ci_str(t, 'recall@10')} | {fmt(mean(values(t, 'mrr@10')))} "
            f"| {n_s} | {ci_str(s, 'recall@10')} |"
        )
    out += ["", "Paired differences (b − a), text-layer questions, 95% CI:", ""]
    pairs = [
        ("hybrid vs BM25", "bm25", "hybrid", "recall@10"),
        ("hybrid vs dense", "dense", "hybrid", "recall@10"),
        ("hybrid vs dense", "dense", "hybrid", "mrr@10"),
        ("hybrid vs BM25", "bm25", "hybrid", "mrr@10"),
        ("rerank vs hybrid", "hybrid", "baseline", "recall@10"),
        ("rerank vs hybrid", "hybrid", "baseline", "recall@1"),
        ("no prefix vs prefix", "baseline", "no_prefix", "recall@10"),
        ("modernbert vs gte-small", "baseline", "modernbert", "recall@10"),
    ]
    for label, a, b, col in pairs:
        out.append(f"- {label}, {col}: {diff_ci(r, a, b, col, 'text_layer')}")
    out += ["", "All questions, recall@10:", ""]
    for label, name in [
        ("no OCR", "baseline"),
        ("vision-LLM OCR", "baseline_vlm"),
        ("corpus scope", "corpus_vlm"),
    ]:
        out.append(f"- {label}: {ci_str(r.df(name), 'recall@10')}")
    out.append(
        "- corpus vs doc scope: "
        f"{diff_ci(r, 'baseline_vlm', 'corpus_vlm', 'recall@10')}; text-layer "
        f"{fmt(mean(values(r.part('corpus_vlm', 'text_layer'), 'recall@10')))}, "
        f"scanned {fmt(mean(values(r.part('corpus_vlm', 'scanned'), 'recall@10')))}"
    )
    return out + [""]


def models(r: Results) -> list[str]:
    ref = MODEL_ROWS["Claude Haiku 5.5"]
    out = ["## Answer-model comparison (test split, paired vs Haiku)", ""]
    out += [
        "| model | correct [95% CI] | Δ vs Haiku | McNemar | unanswerable | failed "
        "| LLM p50 / p95 s | EUR full / 100 q | unsupported share* |",
        "|---" * 9 + "|",
    ]
    ref_correct = mean(values(r.df(ref), "correct"))
    for label, name in MODEL_ROWS.items():
        df = r.df(name)
        info = r.info(name)
        llm_lat = info.get("latency", {}).get("llm", {})
        cost = (info.get("llm") or {}).get("cost_eur_full")
        answered = df.filter(
            pl.col("status").is_not_null() & (pl.col("status") != "abstained")
        )
        unsupported = (
            (answered["status"] == "unsupported").sum() / answered.height
            if answered.height
            else None
        )
        delta = mean(values(df, "correct")) - ref_correct
        cells = [
            ci_str(df, "correct"),
            "–" if name == ref else f"{100 * delta:+.1f} pts",
            "–" if name == ref else paired(r, ref, name),
            fmt(mean(values(r.part(name, "unanswerable"), "correct"))),
            f"{100 * mean(values(df, 'failed')):.1f}%",
            f"{llm_lat.get('p50_ms', math.nan) / 1000:.1f} / "
            f"{llm_lat.get('p95_ms', math.nan) / 1000:.1f}",
            "–" if cost is None else f"{100 * cost / df.height:.2f}",
            "–" if unsupported is None else f"{unsupported:.2f}",
        ]
        out.append(f"| {label} | " + " | ".join(cells) + " |")
    out += [
        "",
        "\\* share of non-abstained answers whose citations all failed Verify "
        "(status `unsupported`; failed answers have no status and are left out).",
        "",
    ]
    return out


def judge(r: Results) -> list[str]:
    out = ["## Judge vs human labels", ""]
    for labels in sorted((r.dir / "human_labels").glob("*.csv")):
        run = r.dir / labels.stem
        rep = judge_agreement(run, labels)
        out.append(f"Labels `{labels.relative_to(r.dir.parent).as_posix()}`:")
        for field, v in rep.items():
            out.append(
                f"- {field}: {int(v['agree'])}/{int(v['n'])} agree, "
                f"Cohen's κ {v['kappa']:.3f}"
            )
        lab = read_labels(labels)
        per_q = pl.read_csv(run / "per_question.csv", infer_schema_length=0)
        joined = lab.join(per_q.select("qid", "faithful"), on="qid")
        human = [_bool(v) is False for v in joined["human_faithful"].to_list()]
        judge_f = [_bool(v) is False for v in joined["faithful"].to_list()]
        both = sum(h and j for h, j in zip(human, judge_f, strict=True))
        out.append(
            f"- marked unfaithful: human {sum(human)}/{joined.height}, "
            f"judge {sum(judge_f)}/{joined.height}, both {both}"
        )
    return out + [""]


def report(results_dir: Path) -> str:
    r = Results(results_dir)
    lines = ["# Recomputed statistics (scripts/reproduce_stats.py)", ""]
    sections = (split, answers, abstention, retrieval, models, judge)
    for section in (*sections, consistency, provenance):
        lines += section(r)
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    text = report(args.results)
    if isinstance(sys.stdout, io.TextIOWrapper):  # Windows consoles default to cp1252
        sys.stdout.reconfigure(encoding="utf-8")
    print(text)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
