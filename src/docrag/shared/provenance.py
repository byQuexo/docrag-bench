"""Which code produced a result: git commit + "were there uncommitted changes?".

A config hash says WHAT was measured; this says WITH WHICH CODE. Two runs with the
same config hash but different commits can differ (a bug fix in Verify changes
the numbers without touching the config).
"""

import logging
import subprocess
from pathlib import Path

log = logging.getLogger(__name__)

# Only paths that can change results count as "dirty". An edited README or a new
# results/ folder must not mark a run as made from uncommitted code.
RESULT_RELEVANT = ["src", "configs", "pyproject.toml", "uv.lock"]


def _git(root: Path, *args: str) -> str | None:
    try:
        # Argument list, no shell: nothing in the arguments can run as a command.
        out = subprocess.run(
            ["git", *args],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        # No git installed, not a repository (Docker image), no commits yet.
        return None
    return out.stdout.strip()


def code_version(root: Path) -> dict[str, str | bool | None]:
    """``{"commit": sha | None, "dirty": bool | None}`` for the repo at ``root``.

    dirty = uncommitted changes (incl. untracked files) under RESULT_RELEVANT.
    A dirty run's numbers can't be reproduced from the commit alone.
    """
    commit = _git(root, "rev-parse", "HEAD")
    if commit is None:
        return {"commit": None, "dirty": None}
    status = _git(root, "status", "--porcelain", "--", *RESULT_RELEVANT)
    return {"commit": commit, "dirty": None if status is None else bool(status)}


class DirtyTreeError(RuntimeError):
    """A benchmark run was started from code that no commit describes."""


def require_clean(
    code: dict[str, str | bool | None], *, allow_dirty: bool = False
) -> dict[str, str | bool | None]:
    """Stop a benchmark run whose code can't be pinned to a commit.

    Aborts (``DirtyTreeError``) when there are uncommitted result-relevant
    changes, or when the git state is unknown (no git, not a repository), unless
    ``allow_dirty``. With ``allow_dirty`` the run goes ahead with a warning, and
    the returned dict (what goes into run.json) says so: ``"dirty"`` keeps its
    value and ``"allow_dirty": True`` records the override.
    """
    if code.get("dirty") is False:
        return {**code, "allow_dirty": False}
    if code.get("dirty") is None:
        reason = "git state unknown (git missing or not a repository)"
    else:
        reason = "uncommitted changes under " + ", ".join(RESULT_RELEVANT)
    if not allow_dirty:
        raise DirtyTreeError(
            f"{reason}: the results could not be reproduced from commit "
            f"{code.get('commit') or '(none)'}. Commit or stash them first, "
            "or pass --allow-dirty (the run is then marked dirty in run.json)."
        )
    log.warning(
        "--allow-dirty: %s; run.json will record dirty=%s", reason, code.get("dirty")
    )
    return {**code, "allow_dirty": True}
