"""LLM-as-judge for answer correctness and faithfulness, with a Pydantic schema.

Rule-based where possible, LLM only where needed:
* Gold "Not answerable": correct iff the system abstained. No LLM call.
* System abstained on an answerable question: incorrect. No LLM call.
* Otherwise: ask the judge model for {reasoning, correct, faithful}.
"""

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


async def judge(
    proxy: LLMProxy,
    route: str,
    q: EvalQuestion,
    result: AnswerResult,
    temperature: float | None = 0.0,
) -> JudgeVerdict:
    is_abstained = abstained(result)
    if not q.answerable:
        return JudgeVerdict(
            reasoning="rule: unanswerable", correct=is_abstained, faithful=True
        )
    if is_abstained:
        return JudgeVerdict(reasoning="rule: abstained", correct=False, faithful=True)

    quotes = "\n".join(f"- {c.quote}" for c in result.verified_citations) or "(none)"
    user = (
        f"Question: {q.question}\n"
        f"Gold answer: {q.answer}\n"
        f"System answer: {result.answer}\n"
        f"Cited evidence quotes:\n{quotes}"
    )
    messages = [
        Message(role="system", content=load_prompt("judge_v1")),
        Message(role="user", content=user),
    ]
    verdict, _ = await proxy.structured(
        route, messages, JudgeVerdict, temperature=temperature
    )
    return verdict
