# 0008. LLM-as-judge after rules, from another model family, checked against human labels

- **Status:** Accepted
- **Date:** 2026-11-01 (judge in `e1ca9c0`; human labels and agreement in `ef2a510`, `5d6b42b`)

## Context

Free-text answers to 735 questions cannot be graded by string match, and
grading them by hand on every run is not possible. A judge is itself a model
with biases; a score from an unchecked judge is not evidence.

## Decision

- **Rules first:** "Not answerable" questions and abstentions are graded by
  rule; the judge only grades `correct` and `faithful`.
- **Judge:** DeepSeek V4.1 Flash via OpenRouter, temperature 0, strict schema,
  reasoning before the verdict, sees the cited passages in full (`judge_v2`).
  Different family from the answer model (Claude Haiku), against
  self-preference bias. Invalid verdicts after one retry are counted, not hidden.
- **Calibration:** `docrag label-sample` draws 50 judged answers into a blind
  CSV, a human labels them (`docrag label`), and `docrag judge-agreement`
  reports raw agreement and Cohen's κ.

## Alternatives considered

- **Human grading only:** too slow for every run and every ablation.
- **Same family as the answer model:** risks the judge preferring its own style.
- **Trust the judge without labels:** see Consequences; it would have shipped a
  meaningless faithfulness number.

## Consequences

- **+** `correct` is usable: 46/50 agreement, κ 0.77.
- **+** The calibration caught a bad metric: `faithful` agrees at κ 0.06 (chance
  level), so faithfulness is shown with a warning and not used as evidence
  (`b57d30c`, `840fab1`). Earlier, `judge_v1` saw only 5-word quotes and rated
  faithfulness 0.46–0.50; `judge_v2` with full passages: 0.83–0.89.
- **−** One judge model and 50 labels from one human on one run.
- **−** A judge call per question costs money; see [0011](0011-real-judge-in-ci.md).
- **Begründung offen:** why DeepSeek V4.1 Flash specifically. The docs give
  only "a different family than the answerer" (`configs/answer_haiku_vlm_v3.yaml`).

## Evidence

- `src/docrag/eval/judge.py`, `src/docrag/eval/agreement.py`, `src/docrag/prompts/judge_v2.txt`
- `results/human_labels/`, README footnote † and "Judge reliability"
- [Lesson 08](../lessons/08_eval.md)
