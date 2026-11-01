"""Verify: pure code that checks every citation after the LLM answered.

(a) The cited ID must be one of the chunks we actually gave the LLM.
    Catches invented IDs ("[doc3:p99:c7]") - a hallucinated source.
(b) The quote must appear verbatim (after normalization) in that chunk.
    Catches quotes the model paraphrased or made up.

Failed citations are removed. If none survive, the answer is marked
"unsupported" (we still return it, clearly labelled, instead of hiding it).
No extra LLM call: Verify costs ~0 ms and 0 EUR, so it can be turned on/off
on the SAME answers for the eval.

What Verify does NOT prove: that the quote actually supports the claim. A real
quote next to a wrong conclusion passes. That's the job of an NLI/hallucination
model (LettuceDetect, planned as the next Verify stage) or the judge.
"""

import re

from docrag.answer.schema import (
    NOT_ANSWERABLE,
    Answer,
    AnswerResult,
    Citation,
    VerifyStatus,
)
from docrag.shared import tracing
from docrag.shared.ids import parse_chunk_id
from docrag.shared.schema import Chunk
from docrag.shared.text import normalize
from docrag.shared.timing import timed

# Markdown table syntax is formatting, not content: "|Europe|45.5|" should match
# a quote "Europe 45.5".
_MARKUP = re.compile(r"<br>|[|*_`#]+")
_SPACES = re.compile(r"\s+")
_INLINE_ID = re.compile(r"\s?\[([A-Za-z0-9._-]+:p\d+:c\d+)\]")


def _canon(text: str) -> str:
    return _SPACES.sub(" ", _MARKUP.sub(" ", normalize(text))).strip()


def is_abstention(text: str) -> bool:
    return normalize(text).rstrip(".") in {normalize(NOT_ANSWERABLE), "not answerable"}


@timed("verify", kind=tracing.GUARDRAIL)
def verify(question: str, answer: Answer, context: list[Chunk]) -> AnswerResult:
    by_id = {c.id: c for c in context}
    citations: list[Citation] = []
    for ev in answer.evidence:
        chunk = by_id.get(ev.id)
        reason: str | None = None
        if chunk is None:
            reason = "id not in context"
        elif not _canon(ev.quote):
            reason = "empty quote"
        elif _canon(ev.quote) not in _canon(chunk.text):
            reason = "quote not found in chunk"
        try:
            doc_id, page, _ = parse_chunk_id(ev.id)
        except ValueError:
            doc_id, page, reason = "", 0, reason or "malformed id"
        citations.append(
            Citation(
                id=ev.id,
                quote=ev.quote,
                doc_id=doc_id,
                page=page,
                verified=reason is None,
                reason=reason,
            )
        )

    verified_ids = {c.id for c in citations if c.verified}
    # Remove inline markers that point to rejected citations.
    # LEARN: re.sub accepts a function: it's called per match and its return value
    # replaces the match. Here: keep "[id]" only if the id was verified.
    cleaned = _INLINE_ID.sub(
        lambda m: m.group(0) if m.group(1) in verified_ids else "", answer.answer
    ).strip()

    # Count CITATIONS (evidence items), not distinct IDs: one ID cited twice, once
    # with a real quote and once with an invented one, must not be "supported".
    # (The first version compared ID sets and let exactly that through.)
    n_verified = sum(c.verified for c in citations)
    status: VerifyStatus
    if is_abstention(answer.answer):
        status = "abstained"
    elif n_verified == 0:
        status = "unsupported"
    elif n_verified < len(citations):
        status = "partially_supported"
    else:
        status = "supported"

    tracing.annotate(
        {
            "docrag.verify.status": status,
            "docrag.verify.n_citations": len(citations),
            "docrag.verify.n_verified": n_verified,
            # Why citations were rejected: the first thing to look at in Phoenix
            # when an answer comes back "unsupported".
            "docrag.verify.rejected": "; ".join(
                f"{c.id}: {c.reason}" for c in citations if not c.verified
            )
            or None,
        }
    )
    return AnswerResult(
        question=question,
        answer=cleaned,
        raw_answer=answer.answer,
        status=status,
        citations=citations,
        context_ids=list(by_id),
    )
