# NNNN. Title: the decision as a short phrase

- **Status:** Proposed | Accepted | Superseded by [NNNN](NNNN-title.md)
- **Date:** YYYY-MM-DD (the commit that introduced the decision, or the day it was proposed)

## Context

The problem and the forces at play, in two to five sentences. Facts only:
constraints, measurements, what breaks without a decision.

## Decision

What we do, in one or two sentences, active voice ("We use X for Y").

## Alternatives considered

- **Option A:** why not.
- **Option B:** why not.

## Consequences

- **+** what gets better.
- **−** what we pay for it, and what it rules out.

## Evidence

Files, commits, CI runs and result folders that show the decision and its
effect. Rationale that no code, commit or doc states is marked
**Begründung offen** (rationale open) instead of being invented.

A `Proposed` ADR has **Problem**, **Options** (a table with for and against)
and **Open questions for the decision** instead of Context, Decision and
Consequences; once decided, it gets those sections and the status `Accepted`.

Keep it to one page. The lessons in `docs/lessons/` tell the story; an ADR is
the short entry you look up.
