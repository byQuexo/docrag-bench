"""Benchmark runner: one ExperimentConfig -> one folder under results/.

It calls exactly the same Retriever/AnswerService as the API. Output files:

* config.yaml        - the config + its hash (what was measured)
* per_question.csv   - one row per question (for error analysis)
* metrics.csv        - one row per slice: all, text, table, image, unanswerable
* run.json           - latency percentiles, LLM cost, failure counts
"""

import asyncio
import json
import logging
import math
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl
import yaml
from opentelemetry.trace import SpanContext

from docrag.answer.schema import AnswerResult
from docrag.answer.service import AnswerService
from docrag.eval.dataset import EvalQuestion, load_questions, select
from docrag.eval.judge import JudgeVerdict, judge, judged_by_llm
from docrag.eval.metrics import (
    citation_precision,
    mrr_at_k,
    ndcg_at_k,
    ranked_pages,
    recall_at_k,
)
from docrag.ingest.indexer import index_documents
from docrag.ingest.parser import get_parser
from docrag.ingest.pipeline import parse_pdf
from docrag.llm.proxy import LLMProxy
from docrag.shared import timing, tracing
from docrag.shared.cache import Cache
from docrag.shared.concurrency import bounded_gather
from docrag.shared.config import ExperimentConfig
from docrag.shared.provenance import code_version
from docrag.shared.schema import Hit
from docrag.shared.settings import ROOT, Settings
from docrag.store.retriever import Retriever
from docrag.wiring import build_ocr, build_retriever, index_components

log = logging.getLogger(__name__)

# Evidence-type slices from the dataset, plus "no_text_layer": questions about PDFs
# whose text parser found nothing (image-only slides/scans) - the OCR target.
SLICES = ["all", "text", "table", "image", "unanswerable", "no_text_layer"]


def question_row(
    q: EvalQuestion,
    hits: list[Hit],
    result: AnswerResult | None,
    verdict: JudgeVerdict | None,
) -> dict[str, Any]:
    """All per-question numbers. NaN = "not applicable" (excluded from means)."""
    gold = set(q.evidence_pages)
    pages = ranked_pages(hits)
    nan = float("nan")
    row: dict[str, Any] = {
        "qid": q.qid,
        "doc_id": q.doc_id,
        "slices": ",".join(q.slices),
        "answerable": q.answerable,
        "recall@1": recall_at_k(pages, gold, 1),
        "recall@5": recall_at_k(pages, gold, 5),
        "recall@10": recall_at_k(pages, gold, 10),
        "mrr@10": mrr_at_k(pages, gold, 10) if gold else nan,
        "ndcg@10": ndcg_at_k(pages, gold, 10),
        "top_pages": json.dumps(pages[:10]),
        "gold_pages": json.dumps(sorted(gold)),
        # Best (reranked) score: compare with the threshold to see how close an
        # abstention was. The mean per slice is also a cheap retrieval signal.
        "top_score": hits[0].score if hits else nan,
    }
    if result is not None:
        all_cits = result.citations
        row |= {  # LEARN: `|=` merges a dict into another (3.9+), like Object.assign.
            "status": result.status,
            "abstained": result.status == "abstained",
            # Abstained by the rerank gate (no LLM call) vs. by the LLM itself.
            # Without this column the two causes could only be guessed.
            "abstained_before_llm": result.abstained_before_llm,
            "n_citations": len(all_cits),
            "n_verified": len(result.verified_citations),
            # Verify OFF: every citation the LLM produced counts.
            "cit_precision_verify_off": citation_precision(all_cits, gold)
            if gold
            else nan,
            # Verify ON: only citations that survived Verify count.
            "cit_precision_verify_on": (
                citation_precision(result.verified_citations, gold) if gold else nan
            ),
            "answer": result.answer,
        }
    if verdict is not None and result is not None:
        row["correct"] = verdict.correct
        # Faithfulness is only averaged over answers the judge model actually read
        # (answerable + not abstained); rule verdicts carry a placeholder.
        row["faithful"] = verdict.faithful if judged_by_llm(q, result) else nan
    return row


def aggregate(rows: list[dict[str, Any]]) -> pl.DataFrame:
    """Mean of every numeric metric, per slice. NaNs are ignored."""
    df = pl.DataFrame(rows)
    metric_cols = [
        c
        for c in df.columns
        if c
        not in {
            "qid",
            "doc_id",
            "slices",
            "status",
            "answer",
            "top_pages",
            "gold_pages",
        }
    ]
    out = []
    for name in SLICES:
        # Exact tag match: str.contains("text") would also match "no_text_layer".
        tags = pl.col("slices").str.split(",")
        part = df if name == "all" else df.filter(tags.list.contains(name))
        if part.height == 0:
            continue
        row: dict[str, Any] = {"slice": name, "n": part.height}
        for col in metric_cols:
            # fill_nan(None) turns NaN into null, which mean() skips.
            values = part[col].cast(pl.Float64).fill_nan(None)
            mean = values.mean()
            row[col] = None if mean is None else round(float(mean), 4)  # type: ignore[arg-type]
        out.append(row)
    return pl.DataFrame(out)


async def run_experiment(
    config: ExperimentConfig, settings: Settings, cache: Cache
) -> Path:
    timing.reset()
    started = time.perf_counter()
    # Captured at the START: a run takes minutes, and files edited meanwhile
    # (e.g. the next prompt) must not mark this run as made from dirty code.
    code = code_version(ROOT)
    questions = select(
        load_questions(settings.qa_glob), config.dataset, settings.pdf_dir
    )
    doc_ids = sorted({q.doc_id for q in questions})
    log.info(
        "%s: %d questions over %d documents", config.name, len(questions), len(doc_ids)
    )

    retriever = build_retriever(settings, config, cache)
    ocr = build_ocr(settings, config, cache)
    index_documents(
        {d: settings.pdf_dir / f"{d}.pdf" for d in doc_ids},
        retriever.store,
        retriever.embeddings,
        prefix_headings=config.prefix_headings,
        cache=cache,
        ocr=ocr,
    )
    # A vision-LLM OCR engine has its own proxy: its spend is indexing cost,
    # reported separately from the answer/judge calls.
    ocr_proxy: LLMProxy | None = getattr(ocr, "proxy", None)

    proxy = (
        LLMProxy.from_settings(settings, cache) if config.llm or config.judge else None
    )
    try:
        rows = await _evaluate(config, retriever, proxy, questions)
        # Tag questions on documents without a text layer (cached parse -> fast).
        parser = get_parser()
        no_text = {
            d
            for d in doc_ids
            if not parse_pdf(settings.pdf_dir / f"{d}.pdf", d, parser, cache)
        }
        for row in rows:
            if row["doc_id"] in no_text:
                row["slices"] = ",".join(filter(None, [row["slices"], "no_text_layer"]))
    finally:
        if proxy is not None:
            await proxy.aclose()

    index = {
        "collection": retriever.store.collection,
        **index_components(config, retriever.embeddings.embedder),
    }
    run_dir = _write_results(
        config, settings.results_dir, rows, proxy, started, index, code, ocr_proxy
    )
    log.info("results written to %s", run_dir)
    return run_dir


async def _evaluate(
    config: ExperimentConfig,
    retriever: Retriever,
    proxy: LLMProxy | None,
    questions: list[EvalQuestion],
) -> list[dict[str, Any]]:
    def scope(q: EvalQuestion) -> list[str] | None:
        return [q.doc_id] if config.scope == "doc" else None

    # One trace per question (not one per run: a trace with 1000 questions in it
    # is unreadable). docrag.run/config_hash let you filter one run in Phoenix,
    # e.g. "the table questions of run X that came back unsupported".
    def attrs(q: EvalQuestion) -> dict[str, tracing.AttrValue]:
        return {
            "input.value": q.question,
            "docrag.qid": q.qid,
            "docrag.doc_id": q.doc_id,
            "docrag.slices": ",".join(q.slices),
            "docrag.answerable": q.answerable,
            "docrag.run": config.name,
            "docrag.config_hash": config.config_hash(),
        }

    if proxy is None or config.llm is None:
        # Retrieval-only run: no LLM, fast, free.
        rows = []
        for q in questions:
            with tracing.root_span("eval.question", attributes=attrs(q)):
                hits = retriever.retrieve(q.question, scope(q))
            rows.append(question_row(q, hits, None, None))
        return rows

    service = AnswerService(retriever, proxy, config.llm)
    # Remember each question's span so the judge (which runs later, as its own
    # phase) can attach to the same trace.
    question_spans: dict[str, SpanContext] = {}

    async def answer(q: EvalQuestion) -> tuple[AnswerResult, list[Hit]]:
        with tracing.root_span("eval.question", attributes=attrs(q)) as span:
            question_spans[q.qid] = span.get_span_context()
            return await service.answer(q.question, doc_ids=scope(q))

    # Local LLMs handle ~1-2 parallel requests well; APIs could take more.
    answered = await bounded_gather(
        (answer(q) for q in questions), limit=config.llm_concurrency
    )
    if answered.n_failed:
        log.warning(
            "%d of %d answers failed: %s",
            answered.n_failed,
            len(questions),
            answered.errors[:3],
        )

    verdicts: list[JudgeVerdict | None] = [None] * len(questions)
    if config.judge:
        judge_route = config.judge

        async def judge_in_trace(
            q: EvalQuestion, result: AnswerResult, hits: list[Hit]
        ) -> JudgeVerdict:
            parent = question_spans.get(q.qid)
            with tracing.child_of(parent, "judge", kind=tracing.EVALUATOR):
                verdict = await judge(
                    proxy,
                    judge_route,
                    q,
                    result,
                    config.judge_temperature,
                    prompt_version=config.judge_prompt_version,
                    chunks={h.chunk.id: h.chunk.text for h in hits},
                )
                tracing.annotate(
                    {
                        "docrag.correct": verdict.correct,
                        "docrag.faithful": verdict.faithful,
                        "output.value": verdict.reasoning,
                    }
                )
                return verdict

        jobs = [
            (i, judge_in_trace(q, res[0], res[1]))
            for i, (q, res) in enumerate(zip(questions, answered.results, strict=True))
            if res is not None
        ]
        judged = await bounded_gather((coro for _, coro in jobs), limit=4)
        for (i, _), verdict in zip(jobs, judged.results, strict=True):
            verdicts[i] = verdict
        if judged.n_failed:
            log.warning("%d judge calls failed: %s", judged.n_failed, judged.errors[:3])

    rows = []
    for q, res, verdict in zip(questions, answered.results, verdicts, strict=True):
        if res is None:  # failed answer: still record retrieval metrics
            row = question_row(
                q,
                await asyncio.to_thread(retriever.retrieve, q.question, scope(q)),
                None,
                None,
            )
            # A failed answer is a wrong answer for the user. Leaving `correct`
            # empty would silently drop it from the mean (and flatter the score).
            row["failed"] = True
            if config.judge:
                row["correct"] = False
            rows.append(row)
        else:
            result, hits = res
            row = question_row(q, hits, result, verdict) | {"failed": False}
            if config.judge:
                # The JUDGE failed (not the system): `correct` stays empty, which
                # is right for the mean, but it must be visible and counted.
                row["judge_failed"] = verdict is None
            rows.append(row)
    return rows


def _write_results(
    config: ExperimentConfig,
    results_dir: Path,
    rows: list[dict[str, Any]],
    proxy: LLMProxy | None,
    started: float,
    index: dict[str, str],
    code: dict[str, str | bool | None],
    ocr_proxy: LLMProxy | None = None,
) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    run_dir = results_dir / f"{stamp}_{config.name}_{config.config_hash()}"
    run_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(
            {"config_hash": config.config_hash(), **config.model_dump()},
            f,
            sort_keys=False,
        )
    pl.DataFrame(rows).write_csv(run_dir / "per_question.csv")
    metrics = aggregate(rows).with_columns(
        pl.lit(config.name).alias("run"),
        pl.lit(config.config_hash()).alias("config_hash"),
    )
    metrics.write_csv(run_dir / "metrics.csv")

    run_info: dict[str, Any] = {
        "wall_seconds": round(time.perf_counter() - started, 1),
        # Which code produced these numbers. Without it, runs made with older
        # code are indistinguishable from current ones (same config hash).
        "code": code,
        "index": index,
        "latency": timing.snapshot(),
    }
    if proxy is not None:
        run_info["llm"] = proxy.stats_dict()  # answers + judge
    # Only when OCR actually ran: an existing index needs no OCR, and a row of
    # zeros would read as "OCR is free".
    if ocr_proxy is not None and (ocr_proxy.stats.calls or ocr_proxy.stats.cache_hits):
        run_info["ocr_llm"] = ocr_proxy.stats_dict()  # VLM OCR during indexing
    (run_dir / "run.json").write_text(json.dumps(run_info, indent=2), encoding="utf-8")
    return run_dir


def summarize(results_dir: Path) -> str:
    """Combine the newest run per config into one Markdown table (slice = all)."""
    latest: dict[str, Path] = {}
    for metrics_file in sorted(results_dir.glob("*/metrics.csv")):
        name = metrics_file.parent.name.split("_", 1)[1]  # drop the timestamp
        latest[name] = metrics_file  # sorted by timestamp -> newest wins
    if not latest:
        return "(no results yet)"
    frames = [pl.read_csv(p).filter(pl.col("slice") == "all") for p in latest.values()]
    df = pl.concat(frames, how="diagonal_relaxed")
    cols = [
        c
        for c in [
            "run",
            "n",
            "recall@1",
            "recall@5",
            "recall@10",
            "mrr@10",
            "ndcg@10",
            "correct",
            "faithful",
            "cit_precision_verify_off",
            "cit_precision_verify_on",
            "abstained",
        ]
        if c in df.columns
    ]
    df = df.select(cols)
    header = "| " + " | ".join(cols) + " |"
    sep = "|" + "---|" * len(cols)
    lines = [header, sep]
    for row in df.iter_rows():
        cells = [
            "" if v is None or (isinstance(v, float) and math.isnan(v)) else str(v)
            for v in row
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
