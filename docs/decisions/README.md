# Architecture decision records

An ADR is a short record of one decision: the context, what was chosen, what
was rejected, and what it costs. Each one fits on a page and points to the
evidence (code, commits, CI runs, result folders), so every "why X and not Y?"
has a written answer that can be checked.

- **Format:** [MADR](https://adr.github.io/madr/)-style, see [`0000-template.md`](0000-template.md).
- **Dates:** the date of the commit that introduced the decision. The git
  history starts with `e1ca9c0` (2026-11-01), which squashes all earlier work;
  ADRs for decisions made before it say so instead of guessing an earlier date.
- **Status:** `Proposed` (open, options only), `Accepted` (in the code),
  `Superseded` (replaced by a later ADR, which is linked). Accepted ADRs are not
  rewritten when the code changes; a new ADR supersedes them.
- **Open rationale:** where no code, commit or doc states why, the ADR says
  **Begründung offen** (rationale open) instead of inventing a reason.
- **Lessons vs ADRs:** [`docs/lessons/`](../lessons/README.md) explain how and
  why things work, at length; ADRs are the lookup entries and link there.

## Index

| # | Decision | Status | Date |
|---|---|---|---|
