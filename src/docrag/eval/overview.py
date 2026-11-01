"""The data behind the dashboard: the newest run per config, read from results/.

Nothing is hard-coded: every number comes from a run's per_question.csv,
config.yaml and run.json, and every interval from ``stats.doc_bootstrap_ci``.
``docrag summary`` (Markdown) and ``GET /results`` (JSON for the web UI) both
read through here.
"""

import json
import math
from pathlib import Path
from typing import Any

import polars as pl
import yaml

from docrag.eval.agreement import judge_agreement
from docrag.eval.stats import doc_bootstrap_ci


def run_dirs(results_dir: Path) -> list[Path]:
    """Every run folder, oldest first (folder names start with a UTC timestamp)."""
    return sorted(p.parent for p in results_dir.glob("*/metrics.csv"))


def run_name(run_dir: Path) -> str:
    """'20261101T162727_answer_haiku_vlm_v3_776cde4ae38c' -> 'answer_haiku_vlm_v3'."""
    return run_dir.name.split("_", 1)[1].rsplit("_", 1)[0]


def values(df: pl.DataFrame, col: str) -> list[float]:
    """A per-question column as floats: true/false -> 1/0, empty or NaN -> nan."""
    if col not in df.columns:
        return [math.nan] * df.height
    raw = df[col].cast(pl.Utf8).str.to_lowercase()
    raw = raw.replace({"true": "1", "false": "0", "": "nan"}).fill_null("nan")
    return raw.cast(pl.Float64).to_list()


def ci(df: pl.DataFrame, col: str) -> dict[str, float] | None:
    """{mean, lo, hi} with a 95% bootstrap CI over documents; None if not applicable."""
    mean, lo, hi = doc_bootstrap_ci(values(df, col), df["doc_id"].to_list())
    return None if math.isnan(mean) else {"mean": mean, "lo": lo, "hi": hi}


def _mean(df: pl.DataFrame, col: str) -> float | None:
    v = [x for x in values(df, col) if not math.isnan(x)]
    return sum(v) / len(v) if v else None


def _run(run_dir: Path, scanned: set[str]) -> dict[str, Any]:
    df = pl.read_csv(run_dir / "per_question.csv", infer_schema_length=0)
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    info = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    has_answers = "correct" in df.columns
    answerable = df.filter(pl.col("answerable").str.to_lowercase() == "true")
    doc = pl.col("doc_id").is_in(list(scanned))
    parts = {"text_layer": df.filter(~doc), "scanned": df.filter(doc)}
    latency = info.get("latency", {})
    return {
        "name": run_name(run_dir),
        "folder": run_dir.name,
        "config_hash": cfg.get("config_hash"),
        "split": (cfg.get("dataset") or {}).get("split"),  # "test" or "dev"
        "commit": (info.get("code") or {}).get("commit"),
        "setup": {
            k: cfg.get(k)
            for k in (
                "retrieval",
                "scope",
                "embedder",
                "ocr",
                "ocr_model",
                "prefix_headings",
                "prompt_version",
                "context_k",
                "llm",
                "judge_prompt_version",
            )
        },
        "n": df.height,
        "recall_at_10": ci(df, "recall@10"),
        "mrr_at_10": _mean(df, "mrr@10"),
        "correct": ci(df, "correct") if has_answers else None,
        "correct_unanswerable": _mean(
            df.filter(pl.col("answerable").str.to_lowercase() == "false"), "correct"
        )
        if has_answers
        else None,
        "abstained": _mean(df, "abstained") if has_answers else None,
        "abstained_answerable": _mean(answerable, "abstained") if has_answers else None,
        # Share of answerable questions stopped by the rerank gate (before the LLM).
        "gate_answerable": _mean(answerable, "abstained_before_llm"),
        "faithful": _mean(df, "faithful") if has_answers else None,
        "citation_precision": _mean(df, "cit_precision_verify_on"),
        **{
            part: {
                "questions": sub.height,
                "recall_at_10": _mean(sub, "recall@10"),
                "correct": _mean(sub, "correct") if has_answers else None,
            }
            for part, sub in parts.items()
        },
        "cost_eur_full": (info.get("llm") or {}).get("cost_eur_full"),
        "latency_p50_ms": {
            k: latency[k]["p50_ms"] for k in ("answer", "rerank", "llm") if k in latency
        },
    }


def _clean(obj: Any) -> Any:
    """NaN -> None, recursively: JSON has no NaN."""
    if isinstance(obj, float):
        return None if math.isnan(obj) else obj
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean(v) for v in obj]
    return obj


def overview(results_dir: Path) -> dict[str, Any]:
    """Newest run per config name, plus split facts and judge-vs-human agreement."""
    # run_dirs is oldest first, so the newest run of each name wins.
    latest = {run_name(d): d for d in run_dirs(results_dir)}
    if not latest:
        return {"split": None, "runs": [], "judge_agreement": []}
    # Scanned documents are tagged in newer runs; apply the tag to every run.
    scanned: set[str] = set()
    frames = {}
    for name, d in latest.items():
        frames[name] = pl.read_csv(d / "per_question.csv", infer_schema_length=0)
        tagged = frames[name].filter(pl.col("slices").str.contains("no_text_layer"))
        scanned |= set(tagged["doc_id"].to_list())
    biggest = max(frames.values(), key=lambda f: f.height)
    agreement = []
    for labels in sorted((results_dir / "human_labels").glob("*.csv")):
        run = results_dir / labels.stem
        if (run / "per_question.csv").exists():
            agreement.append(
                {"run": run_name(run), "labels": judge_agreement(run, labels)}
            )
    result: dict[str, Any] = _clean(
        {
            "split": {
                "questions": biggest.height,
                "documents": biggest["doc_id"].n_unique(),
                "scanned_documents": len(scanned & set(biggest["doc_id"].to_list())),
            },
            "runs": [_run(d, scanned) for d in latest.values()],
            "judge_agreement": agreement,
        }
    )
    return result
