# 0009. Benchmark runs refuse to start from a dirty working tree

- **Status:** Accepted
- **Date:** 2026-11-09 (`3fa3e59`)

## Context

`run.json` already recorded the git commit and a `dirty` flag, but nothing
stopped a run from uncommitted code. The headline `answer_v3` test run recorded
commit `826f887` with `dirty: true`, so its exact code was unknown.

## Decision

`docrag eval` aborts before loading questions or calling any API when `src/`,
`configs/`, `pyproject.toml` or `uv.lock` have uncommitted or untracked
changes, or when the git state is unknown. `--allow-dirty` overrides it, logs a
warning and records `"allow_dirty": true` in `run.json`.

## Alternatives considered

- **Record the flag only (the previous state):** it documented the problem
  after the fact; the headline number was already unreproducible.
- **Check the whole tree:** notes, lessons or results files would block runs
  that cannot change a number.

## Consequences

- **+** Every new result folder names a commit that can be checked out and re-run.
  The headline run was re-run clean on 2026-11-15 (commit `251a50d`,
  `dirty: false`, correct 0.546; 21 of 735 rows differ from the first run
  because GPU rerank scores of near-tied chunks vary).
- **−** Experimenting needs a commit (or the explicit flag) before each run.
- **−** A clean commit is not full reproducibility: the LLM cache is local and
  hosted models can drift.

## Evidence

- `src/docrag/shared/provenance.py`, `src/docrag/eval/runner.py`, `tests/unit/test_provenance.py`
- `results/20261115T171818_answer_haiku_vlm_v3_776cde4ae38c/run.json`
- README "How far to trust these numbers", [lesson 08](../lessons/08_eval.md)
