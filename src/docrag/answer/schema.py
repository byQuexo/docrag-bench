"""What the LLM must return (Answer) and what the API returns (AnswerResult)."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

NOT_ANSWERABLE = "Not answerable"


class Evidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(description="Chunk ID from the context, e.g. doc3:p2:c1")
    quote: str = Field(description="Verbatim text copied from that chunk")


class Answer(BaseModel):
    """The structured LLM output, enforced via response_format."""

    model_config = ConfigDict(extra="forbid")

    answer: str = Field(
        description="The answer with inline citations like [doc3:p2:c1]"
    )
    evidence: list[Evidence]


VerifyStatus = Literal["supported", "partially_supported", "unsupported", "abstained"]


class Citation(BaseModel):
    id: str
    quote: str
    doc_id: str
    page: int
    verified: bool
    reason: str | None = None  # why verification failed, if it did


class AnswerResult(BaseModel):
    question: str
    answer: str  # after Verify (inline IDs of removed citations stripped)
    raw_answer: str  # exactly what the LLM said (for the "Verify off" eval axis)
    status: VerifyStatus
    citations: list[Citation]  # ALL citations, each marked verified or not
    context_ids: list[str]  # chunks that were shown to the LLM
    abstained_before_llm: bool = False  # nothing passed the rerank threshold
    # Usage of the answer call (0 when abstained before the LLM). Cost is what
    # the call costs uncached, priced from its usage (shown per answer in the UI).
    input_tokens: int = 0
    output_tokens: int = 0
    cost_eur: float = 0.0

    @property
    def verified_citations(self) -> list[Citation]:
        return [c for c in self.citations if c.verified]
