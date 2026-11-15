# 0012. One version source: uv lockfile, mypy strict everywhere, pre-commit through `uv run`

- **Status:** Accepted
- **Date:** 2026-11-01 (uv and mypy strict on `src`, in `e1ca9c0`); extended 2026-11-14 (`914e4e5`)

## Context

The project runs on Windows locally, on Linux in CI and in Docker. A check
that passes locally and fails in CI (or the reverse) wastes a round trip and
erodes trust in the checks.

## Decision

- **uv** manages Python 3.12, the venv and a committed `uv.lock`; CI and the
  image install with `--locked`.
- **mypy `--strict`** covers `src`, `tests` and `scripts`, with the pydantic
  plugin (`init_typed`, `init_forbid_extra`) so model constructors are typed.
- **ruff** with `ASYNC`, `RUF`, `SIM`, `PT` on top of the basics; the two
  disabled rules carry a written reason in `pyproject.toml`.
- **pre-commit** runs ruff, mypy and `uv lock --check` as local hooks through
  `uv run`, i.e. with the locked versions.

## Alternatives considered

- **pip + venv + pip-tools / Poetry:** several tools or a slower resolver; uv
  also installs Python itself.
- **The `ruff-pre-commit` / `mirrors-mypy` repo hooks:** they pin their own
  versions, which drift from `uv.lock`, so a commit passes locally and fails in CI.
- **mypy strict on `src` only:** tests built models from untyped `**kwargs`;
  turning on `init_typed` found a real `str` vs `Literal` bug in the Qdrant store.

## Consequences

- **+** One version of every tool everywhere; CI reproduces local results.
- **+** Type errors in tests and scripts are caught before they run.
- **−** `uv add` / `uv remove` re-sync without the optional extras (`gpu`,
  `demo`) unless told otherwise; a local pitfall, not a CI one.
- **−** mypy on the whole project in a hook is slower than per-file hooks.

## Evidence

- `pyproject.toml`, `uv.lock`, `.pre-commit-config.yaml`, `.github/workflows/ci.yml` (job `lint-test`)
- [Lesson 00](../lessons/00_setup_and_tooling.md), [lesson 14, section 1](../lessons/14_operability.md)
