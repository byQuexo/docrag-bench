# 0019. Integration tests start Qdrant with testcontainers

- **Status:** Accepted
- **Date:** 2026-11-14 (`540d4b9`)

## Context

Unit tests use fakes and `:memory:` Qdrant; they cannot catch server-side
behaviour such as payload serialisation, query plans or HTTP errors. The
integration tests previously relied on a Qdrant `services:` container that
existed only in CI.

## Decision

`tests/e2e` start `qdrant/qdrant:v1.20.0` through testcontainers (skipped when
no Docker is available). Unit and integration tests run in one `pytest` call
with line and branch coverage; CI fails below a floor of 80%.

## Alternatives considered

- **CI `services:` container (the previous setup):** works only in CI; locally
  the tests needed a hand-started Qdrant, and state could leak between runs.
- **Separate jobs for unit and integration tests:** coverage would count only
  one of them.

## Consequences

- **+** The same `uv run pytest` runs on a laptop and in CI; each run gets a
  fresh server, removed even when a test fails.
- **+** In CI: 281 tests pass, 6 of them against a real Qdrant; coverage 86.3%.
  The floor sits a few points below, so removing tests fails, noise does not.
- **−** Needs Docker on the machine; without it the integration tests are skipped.
- **−** Slower than unit tests (container start).

## Evidence

- `tests/e2e/test_e2e.py`, `pyproject.toml` (`[tool.coverage.*]`), `.github/workflows/ci.yml` (job `lint-test`)
- [Lesson 09](../lessons/09_testing_ci_docker.md), [lesson 14, section 8](../lessons/14_operability.md)
