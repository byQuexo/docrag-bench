# 0005. Citations are verified in code, not by an LLM

- **Status:** Accepted
- **Date:** 2026-11-01 (predates the git history; in `e1ca9c0`)

## Context

The answer cites chunks with ids and short quotes. An LLM can invent a chunk
id, misquote a passage, or cite a chunk it never saw. A citation the user
clicks must lead to text that really is there.

## Decision

After the LLM answers, `verify.py` checks every citation in code: the chunk id
must be in the context sent to the model, and the normalised quote must be a
substring of that chunk's raw text. Failed citations are kept with a reason,
their inline markers are removed, and the answer gets a status (`supported`,
`partially_supported`, `unsupported`). The API returns only verified sources.

## Alternatives considered

- **Ask the LLM (or a second LLM) to check itself:** costs a call, adds
  latency, is not deterministic and cannot be unit-tested exactly.
- **Trust the citations:** what most RAG answers do; the user cannot tell
  which sentence came from where.

## Consequences

- **+** 0 ms, 0 EUR, deterministic, testable. Citation precision can be
  measured on verified citations (0.789 in the headline run).
- **+** "unsupported" is shown, not hidden.
- **−** It proves the quote exists, not that it supports the claim: a real
  quote next to a wrong conclusion passes. Short quotes (one number) pass easily.
  Semantic support would need an entailment model (LettuceDetect, future work).
- **−** Normalisation can miss legitimate matches ("%" vs "percent").

## Evidence

- `src/docrag/answer/verify.py`, `src/docrag/answer/schema.py`
- [Lesson 06](../lessons/06_answer_and_verify.md), [interview prep 20, 24](../lessons/10_interview_prep.md)
