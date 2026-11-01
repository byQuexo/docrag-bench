"""The judge: rules first, and judge_v2 sees full cited passages, not just quotes."""

from docrag.answer.schema import AnswerResult, Citation
from docrag.eval.dataset import EvalQuestion
from docrag.eval.judge import judge, render_cited
from docrag.llm.fake import FakeProvider
from docrag.llm.proxy import LLMProxy

PASSAGE = "Revenue in Europe was 45.5 million euros, up 12 percent from 2023."


def _q(answer: str = "45.5 million euros") -> EvalQuestion:
    return EvalQuestion(
        qid="q",
        doc_id="d",
        question="What was the revenue in Europe?",
        answer=answer,
        answer_format="Str",
        evidence_pages=[2],
        slices=["text"],
    )


def _result(*cits: tuple[str, str, bool], answer: str = "45.5 million") -> AnswerResult:
    return AnswerResult(
        question="q",
        answer=answer,
        raw_answer=answer,
        status="supported",
        citations=[
            Citation(id=i, quote=qt, doc_id="d", page=2, verified=ok)
            for i, qt, ok in cits
        ],
        context_ids=[],
    )


def test_render_cited_shows_full_text_of_verified_citations_only() -> None:
    result = _result(
        ("d:p2:c0", "45.5 million euros", True),
        ("d:p2:c0", "up 12 percent", True),
        ("d:p9:c3", "invented", False),
    )
    text = render_cited(result, {"d:p2:c0": PASSAGE, "d:p9:c3": "other"})
    assert PASSAGE in text
    assert text.count("[d:p2:c0]") == 1  # one block per chunk, both quotes listed
    assert '"45.5 million euros"; "up 12 percent"' in text
    assert "d:p9:c3" not in text  # rejected citation: not a source


def test_render_cited_without_sources() -> None:
    assert render_cited(_result(), {}) == "(none)"


async def test_v2_prompt_contains_passages() -> None:
    provider = FakeProvider()
    proxy = LLMProxy({"fake": provider})
    result = _result(("d:p2:c0", "45.5 million", True))
    verdict = await judge(proxy, "fake:j", _q(), result, chunks={"d:p2:c0": PASSAGE})
    assert verdict.correct
    system, user = provider.calls[0]["messages"]
    assert "FULL TEXT" in system.text  # judge_v2.txt
    assert PASSAGE in user.text


async def test_v1_prompt_is_unchanged() -> None:
    # Old runs must stay reproducible from the cache: same messages as before.
    provider = FakeProvider()
    proxy = LLMProxy({"fake": provider})
    result = _result(("d:p2:c0", "45.5 million", True))
    await judge(proxy, "fake:j", _q(), result, prompt_version="judge_v1")
    user = provider.calls[0]["messages"][1].text
    assert user.endswith("Cited evidence quotes:\n- 45.5 million")
    assert PASSAGE not in user


async def test_rules_need_no_llm() -> None:
    provider = FakeProvider()
    proxy = LLMProxy({"fake": provider})
    abstained = _result(answer="Not answerable").model_copy(
        update={"status": "abstained"}
    )
    v = await judge(proxy, "fake:j", _q("Not answerable"), abstained)
    assert v.correct and provider.calls == []
