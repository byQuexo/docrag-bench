"""Which code produced a result: git commit + "were there uncommitted changes?".

A config hash says WHAT was measured; this says WITH WHICH CODE. Two runs with the
same config hash but different commits can differ (a bug fix in Verify changes
the numbers without touching the config).
"""

import subprocess
from pathlib import Path

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
