# 0011. A real judge in CI with a spend cap; fork PRs report the judge as skipped

- **Status:** Accepted
- **Date:** 2026-11-14 (`312de63`)

## Context

The gate ([0010](0010-ci-eval-gate.md)) needs a correctness signal. GitHub gives
no secrets to pull requests from forks, so they cannot call a paid API.

## Decision

- **Real judge:** CI calls the same DeepSeek judge as the benchmark, with the
  OpenRouter key as a repository secret.
- **Spend cap:** `DOCRAG_BUDGET_EUR=0.50` per run, plus a credit limit on the
  key at OpenRouter.
- **Fork PRs:** run the retrieval half only (`configs/ci_gate_retrieval.yaml`);
  the judge check prints SKIPPED, never PASSED.
- **Missing key on the main repo:** the job fails instead of degrading.

## Alternatives considered

- **A fake judge:** always says "correct", so the gate would be a fake gate
  (the deciding argument, 2026-11-14).
- **Retrieval-only gate everywhere:** cheaper, but a broken prompt or answer
  path would pass.
- **Skipping the job silently on forks:** a green check would look like a pass.

## Consequences

- **+** The gate measures what users get: answers, not just retrieval.
- **+** Spend is bounded twice; a cold run costs a few cents, a warm one less.
- **−** Fork PRs are not judged; a maintainer must re-run them from a branch.
- **−** CI depends on OpenRouter and DeepSeek being available.

## Evidence

- `.github/workflows/ci.yml` (step "Choose gate config"), `configs/ci_gate_retrieval.yaml`,
  `scripts/eval_gate.py` (`--require-judge`), `tests/unit/test_eval_gate.py`
- [Lesson 14, section 7](../lessons/14_operability.md)
