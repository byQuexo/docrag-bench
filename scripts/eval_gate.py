"""CI quality gate: fail the build when retrieval or answer quality drops.

    uv run python scripts/eval_gate.py <results run dir> [--require-judge]

Reads the run's metrics.csv (row slice=all) and compares it with the thresholds
in fixtures/ci_gate.yaml. Prints a Markdown table (also appended to the GitHub
job summary when $GITHUB_STEP_SUMMARY is set) and exits 1 if any check fails.

Checks:
* the run evaluated exactly the expected number of questions (a broken download
  or filter must not "pass" on 3 easy questions);
* recall@10 >= threshold (retrieval);
* correct >= threshold: share of questions the LLM judge marked correct, failed
  answers counted as wrong (answer quality);
* the judge itself failed on at most 10% of questions, otherwise its score is
  not a measurement.

Without judge columns (retrieval-only run for fork PRs) the answer check is
SKIPPED, unless --require-judge makes that a failure.
"""

import argparse
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl
import yaml

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "fixtures" / "ci_gate.yaml"
MAX_JUDGE_FAILED = 0.10


@dataclass
class Check:
    name: str
    value: float | None
    threshold: float | None
    status: str  # PASS, FAIL, SKIPPED

    def row(self) -> str:
        def fmt(x: float | None) -> str:
            return "-" if x is None else f"{x:.3f}"

        cells = [self.name, fmt(self.value), fmt(self.threshold), self.status]
        return "| " + " | ".join(cells) + " |"


def _value(row: dict[str, Any], col: str) -> float | None:
    v = row.get(col)
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    return float(v)


def _at_least(name: str, value: float | None, threshold: float | None) -> Check:
    if value is None:
        return Check(name, None, threshold, "SKIPPED")
    if threshold is None:
        return Check(name, value, None, "FAIL (no threshold set)")
    return Check(name, value, threshold, "PASS" if value >= threshold else "FAIL")


def evaluate(
    run_dir: Path, spec: dict[str, Any], require_judge: bool = False
) -> list[Check]:
    metrics = pl.read_csv(run_dir / "metrics.csv")
    row = metrics.filter(pl.col("slice") == "all").row(0, named=True)
    thresholds = spec["thresholds"]

    n = int(row["n"])
    expected = int(spec["expected_questions"])
    checks = [
        Check(
            "questions evaluated",
            n,
            expected,
            "PASS" if n == expected else "FAIL",
        ),
        _at_least("recall@10", _value(row, "recall@10"), thresholds.get("recall@10")),
    ]
    correct = _at_least(
        "correct (LLM judge)", _value(row, "correct"), thresholds.get("correct")
    )
    if correct.status == "SKIPPED" and require_judge:
        correct.status = "FAIL (no judge results)"
    checks.append(correct)

    judge_failed = _value(row, "judge_failed")
    if judge_failed is not None:
        ok = judge_failed <= MAX_JUDGE_FAILED
        checks.append(
            Check(
                "judge failures (max)",
                judge_failed,
                MAX_JUDGE_FAILED,
                "PASS" if ok else "FAIL",
            )
        )
    return checks


def report(run_dir: Path, checks: list[Check]) -> str:
    lines = [
        f"## Eval quality gate: {run_dir.name}",
        "",
        "| check | value | threshold | status |",
        "|---|---|---|---|",
        *(c.row() for c in checks),
        "",
    ]
    return "\n".join(lines)


def passed(checks: list[Check]) -> bool:
    return all(c.status in ("PASS", "SKIPPED") for c in checks)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--require-judge", action="store_true")
    args = parser.parse_args()
    with open(SPEC, encoding="utf-8") as f:
        spec = yaml.safe_load(f)
    checks = evaluate(args.run_dir, spec, require_judge=args.require_judge)
    text = report(args.run_dir, checks)
    print(text)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as f:
            f.write(text + "\n")
    if not passed(checks):
        sys.exit("eval gate FAILED")
    print("eval gate passed")


if __name__ == "__main__":
    main()
