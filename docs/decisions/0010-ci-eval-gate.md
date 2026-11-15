# 0010. CI eval gate on fixed dev-split questions, thresholds with a margin

- **Status:** Accepted
- **Date:** 2026-11-14 (`312de63`)

## Context

Unit tests prove the code does what it says; they cannot prove the answers are
still good. A retrieval bug or a broken prompt can pass every test. A full
benchmark run takes too long and costs too much for every push.

## Decision

The `eval-gate` CI job runs the real pipeline (ONNX models, Claude Haiku
answers, DeepSeek judge) on a fixed set: 4 small PDFs, 38 questions from the
**dev** split, dataset pinned to one Hugging Face revision. It fails when
recall@10 < 0.65, the judged share correct < 0.40, judge failures > 10%, or
fewer questions than expected were evaluated. The LLM cache is restored
between runs. Results are uploaded as an artifact.

## Alternatives considered

- **Test split:** anything checked on every push becomes a tuning signal; test
  must stay for the reported number ([0007](0007-document-split-and-statistics.md)).
- **Thresholds at the measured value:** with 38 questions, one question moves
  `correct` by 2.6 points; the gate would flap on noise.
- **No cache in CI:** every push would pay for every call and see answer
  variance. Keys are content hashes ([0006](0006-own-llm-proxy.md)), so changed
  inputs miss the cache and are measured fresh.

## Consequences

- **+** Breakage fails the build. Measured in CI (run 43730701046, `88bfed9`):
  recall@10 0.790 [0.670, 0.909], correct 0.579 [0.463, 0.714], 0 judge failures.
  The thresholds sit below both CI and a local GPU run (0.748 / 0.500).
- **+** A broken download cannot pass on three easy questions.
- **−** Catches breakage, not one-point drifts; small differences belong in the
  benchmark with CIs.
- **−** Provider-side model drift is not caught on cache hits.

## Evidence

- `.github/workflows/ci.yml` (job `eval-gate`), `configs/ci_gate.yaml`,
  `fixtures/ci_gate.yaml`, `scripts/eval_gate.py`, `scripts/download_gate_data.py`
- [CI run 43730701046](https://github.com/byQuexo/docrag-bench/actions/runs/43730701046)
- [Lesson 14, section 7](../lessons/14_operability.md)
