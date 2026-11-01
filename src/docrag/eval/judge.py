"""LLM-as-judge for answer correctness and faithfulness, with a Pydantic schema.

Rule-based where possible, LLM only where needed:
* Gold "Not answerable": correct iff the system abstained. No LLM call.
* System abstained on an answerable question: incorrect. No LLM call.
* Otherwise: ask the judge model for {reasoning, correct, faithful}.
"""

from collections.abc import Mapping

from pydantic import BaseModel, ConfigDict

from docrag.answer.context import load_prompt
from docrag.answer.schema import AnswerResult
from docrag.answer.verify import is_abstention
from docrag.eval.dataset import EvalQuestion
from docrag.llm.proxy import LLMProxy
from docrag.llm.types import Message


class JudgeVerdict(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # reasoning FIRST: the model writes its justification before committing to
    # the booleans (field order = generation order with structured output).
    reasoning: str
    correct: bool
    faithful: bool


def abstained(result: AnswerResult) -> bool:
    return result.status == "abstained" or is_abstention(result.answer)


def judged_by_llm(q: EvalQuestion, result: AnswerResult) -> bool:
    """True if the judge MODEL read this answer (vs. a rule deciding it).

    Only then is `faithful` a measurement. The rules set faithful=True as a
    placeholder ("an abstention claims nothing"), which must not be averaged:
    with ~50% abstentions it inflated the faithfulness score.
    """
    return q.answerable and not abstained(result)


def render_cited(result: AnswerResult, chunks: Mapping[str, str]) -> str:
    """Full text of every passage behind a VERIFIED citation (what the user sees).

    One block per chunk ID, with the quote(s) the system picked from it.
    """
    quotes: dict[str, list[str]] = {}
    for c in result.verified_citations:
        quotes.setdefault(c.id, []).append(c.quote)
    if not quotes:
        return "(none)"
    blocks = []
    for cid, qs in quotes.items():
        picked = "; ".join(f'"{q}"' for q in qs)
        # Fallback to the quote if the chunk text is missing (shouldn't happen:
        # verified citations always point at a chunk that was in the context).
        text = chunks.get(cid, " ".join(qs))
        blocks.append(f"[{cid}] (quoted: {picked})\n{text}")
    return "\n\n".join(blocks)


async def judge(
    proxy: LLMProxy,
    route: str,
    q: EvalQuestion,
    result: AnswerResult,
    temperature: float | None = 0.0,
    *,
    prompt_version: str = "judge_v2",
    chunks: Mapping[str, str] | None = None,
) -> JudgeVerdict:
    """Grade one answer. ``chunks``: chunk id -> text of the answer's context."""
    is_abstained = abstained(result)
    if not q.answerable:
        return JudgeVerdict(
            reasoning="rule: unanswerable", correct=is_abstained, faithful=True
        )
    if is_abstained:
        return JudgeVerdict(reasoning="rule: abstained", correct=False, faithful=True)

    if prompt_version == "judge_v1":
        # Kept byte-identical so old runs stay reproducible from the cache.
        quotes = (
            "\n".join(f"- {c.quote}" for c in result.verified_citations) or "(none)"
        )
        evidence = f"Cited evidence quotes:\n{quotes}"
    else:
        evidence = f"Cited passages:\n{render_cited(result, chunks or {})}"
    user = (
        f"Question: {q.question}\n"
        f"Gold answer: {q.answer}\n"
        f"System answer: {result.answer}\n"
        f"{evidence}"
    )
    messages = [
        Message(role="system", content=load_prompt(prompt_version)),
        Message(role="user", content=user),
    ]
    verdict, _ = await proxy.structured(
        route, messages, JudgeVerdict, temperature=temperature
    )
    return verdict
