"""scripts/reproduce_stats.py recomputes the README's numbers from results/.

Pins the recomputation against the committed run artifacts: if a result file or
the statistics code changes, this fails and the README must be checked again.
"""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_report_recomputes_headline_numbers_from_results() -> None:
    spec = importlib.util.spec_from_file_location(
        "reproduce_stats", ROOT / "scripts" / "reproduce_stats.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    text = module.report(ROOT / "results")
    assert "735 questions over 93 documents" in text
    assert "| + prompt answer_v3, 10 chunks | 0.544 [0.501, 0.590] |" in text
    assert "v3 vs vlm, all: +45/−19, McNemar p = 0.002" in text
    assert "faithful: 40/50 agree, Cohen's κ 0.057" in text
    assert "| 826f887 | True |" in text  # the headline run's code was dirty
