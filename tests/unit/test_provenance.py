"""Dirty-tree guard: a benchmark run must be attributable to a commit.

Clean tree -> runs. Uncommitted result-relevant changes -> aborts before any work.
``--allow-dirty`` -> runs with a warning, and run.json records the override.
"""

import argparse
import logging
import subprocess
from pathlib import Path
from typing import Any

import pytest

from docrag import cli
from docrag.eval import runner
from docrag.shared.config import ExperimentConfig
from docrag.shared.provenance import DirtyTreeError, code_version, require_clean

CONFIG = Path(__file__).resolve().parents[2] / "configs" / "e2e_fake.yaml"


def _repo(tmp_path: Path) -> Path:
    def git(*args: str) -> None:
        subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True)

    git("init", "-q")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("x = 1\n")
    git("add", ".")
    git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "init")
    return tmp_path


def test_clean_tree_runs_without_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    code = code_version(_repo(tmp_path))
    with caplog.at_level(logging.WARNING):
        recorded = require_clean(code)
    assert recorded["dirty"] is False and recorded["allow_dirty"] is False
    assert not caplog.records


def test_dirty_tree_aborts_by_default(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    (repo / "src" / "a.py").write_text("x = 2\n")  # uncommitted code change
    with pytest.raises(DirtyTreeError, match="--allow-dirty"):
        require_clean(code_version(repo))


def test_untracked_config_counts_as_dirty(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    (repo / "configs").mkdir()
    (repo / "configs" / "new.yaml").write_text("name: new\n")  # never committed
    with pytest.raises(DirtyTreeError):
        require_clean(code_version(repo))


def test_unknown_git_state_aborts_by_default(tmp_path: Path) -> None:
    with pytest.raises(DirtyTreeError, match="unknown"):
        require_clean(code_version(tmp_path))  # not a repository


def test_dirty_tree_with_allow_dirty_warns_and_is_recorded(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    repo = _repo(tmp_path)
    (repo / "src" / "a.py").write_text("x = 2\n")
    with caplog.at_level(logging.WARNING):
        recorded = require_clean(code_version(repo), allow_dirty=True)
    assert recorded["dirty"] is True and recorded["allow_dirty"] is True
    assert len(str(recorded["commit"])) == 40
    assert "allow-dirty" in caplog.text


async def test_run_experiment_aborts_before_any_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        runner, "code_version", lambda _: {"commit": "abc", "dirty": True}
    )

    def must_not_run(*_: Any) -> None:
        raise AssertionError("questions loaded although the tree is dirty")

    monkeypatch.setattr(runner, "load_questions", must_not_run)
    config = ExperimentConfig.from_yaml(CONFIG)
    with pytest.raises(DirtyTreeError):
        await runner.run_experiment(config, None, None)  # type: ignore[arg-type]


@pytest.mark.parametrize("flag", [[], ["--allow-dirty"]])
def test_cli_passes_allow_dirty(
    monkeypatch: pytest.MonkeyPatch, flag: list[str]
) -> None:
    seen: list[bool] = []

    async def fake_run(*_: Any, allow_dirty: bool) -> Path:
        seen.append(allow_dirty)
        return Path("results/x")

    monkeypatch.setattr(runner, "run_experiment", fake_run)
    monkeypatch.setattr(runner, "summarize", lambda _: "")
    monkeypatch.setattr(cli, "Cache", lambda _: None)
    cli.main(["eval", *flag, str(CONFIG)])
    assert seen == [bool(flag)]


def test_cli_exits_with_message_on_dirty_tree(monkeypatch: pytest.MonkeyPatch) -> None:
    async def dirty_run(*_: Any, allow_dirty: bool) -> Path:
        raise DirtyTreeError("uncommitted changes")

    monkeypatch.setattr(runner, "run_experiment", dirty_run)
    monkeypatch.setattr(cli, "Cache", lambda _: None)
    with pytest.raises(SystemExit, match="refusing to run"):
        cli._cmd_eval(argparse.Namespace(configs=[CONFIG], allow_dirty=False))
