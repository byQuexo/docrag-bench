"""scripts/eval_gate.py: the CI quality gate's pass/fail logic."""

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any

import polars as pl
import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def gate() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "eval_gate", ROOT / "scripts" / "eval_gate.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SPEC: dict[str, Any] = {
    "expected_questions": 38,
    "thresholds": {"recall@10": 0.70, "correct": 0.50},
}


def _run(tmp_path: Path, **cols: Any) -> Path:
    row = {"slice": "all", "n": 38, "recall@10": 0.80, **cols}
    pl.DataFrame([row]).write_csv(tmp_path / "metrics.csv")
    return tmp_path


def _statuses(checks: list[Any]) -> dict[str, str]:
    return {c.name: c.status for c in checks}


def test_passes_above_thresholds(gate: ModuleType, tmp_path: Path) -> None:
    run = _run(tmp_path, correct=0.60, judge_failed=0.0)
    checks = gate.evaluate(run, SPEC)
    assert gate.passed(checks)
    assert "| recall@10 | 0.800 | 0.700 | PASS |" in gate.report(run, checks)


def test_fails_when_answer_quality_drops(gate: ModuleType, tmp_path: Path) -> None:
    checks = gate.evaluate(_run(tmp_path, correct=0.40, judge_failed=0.0), SPEC)
    assert not gate.passed(checks)
    assert _statuses(checks)["correct (LLM judge)"] == "FAIL"


def test_fails_when_fewer_questions_ran(gate: ModuleType, tmp_path: Path) -> None:
    checks = gate.evaluate(_run(tmp_path, n=3, correct=1.0), SPEC)
    assert _statuses(checks)["questions evaluated"] == "FAIL"


def test_unreliable_judge_fails_the_gate(gate: ModuleType, tmp_path: Path) -> None:
    checks = gate.evaluate(_run(tmp_path, correct=0.9, judge_failed=0.2), SPEC)
    assert _statuses(checks)["judge failures (max)"] == "FAIL"


def test_retrieval_only_run_skips_the_judge_unless_required(
    gate: ModuleType, tmp_path: Path
) -> None:
    run = _run(tmp_path)  # no correct/judge_failed columns
    checks = gate.evaluate(run, SPEC)
    assert gate.passed(checks)
    assert _statuses(checks)["correct (LLM judge)"] == "SKIPPED"
    strict = gate.evaluate(run, SPEC, require_judge=True)
    assert not gate.passed(strict)


def test_missing_threshold_is_a_failure(gate: ModuleType, tmp_path: Path) -> None:
    spec = {**SPEC, "thresholds": {"recall@10": None, "correct": 0.5}}
    checks = gate.evaluate(_run(tmp_path, correct=0.6), spec)
    assert not gate.passed(checks)


def test_committed_spec_matches_the_configs() -> None:
    import yaml

    from docrag.shared.config import ExperimentConfig

    spec = yaml.safe_load((ROOT / "fixtures" / "ci_gate.yaml").read_text("utf-8"))
    full = ExperimentConfig.from_yaml(ROOT / "configs" / "ci_gate.yaml")
    retrieval = ExperimentConfig.from_yaml(ROOT / "configs" / "ci_gate_retrieval.yaml")
    # Fork PRs must measure the SAME retrieval as the full gate.
    skip = {
        "name",
        "llm",
        "judge",
        "llm_temperature",
        "judge_temperature",
        "llm_concurrency",
    }
    assert full.model_dump(exclude=skip) == retrieval.model_dump(exclude=skip)
    assert full.dataset.split == "dev"  # never gate (= tune) on the test split
    assert len(spec["documents"]) == 4
