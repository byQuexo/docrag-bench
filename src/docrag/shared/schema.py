"""Data contracts shared by every stage.

Parser -> list[Block] -> Chunker -> list[Chunk] -> Store -> list[Hit].

These are the "types at the seams". If a stage changes internally but still
produces valid Blocks/Chunks/Hits, nothing else has to change.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

BlockType = Literal["text", "table", "image"]
# Where the text came from. "ocr"/"vlm" text is only as trustworthy as the model
# that read it, so the eval reports it separately.
Origin = Literal["text", "ocr", "vlm"]


class Block(BaseModel):
    """One typed piece of a page, as produced by a parser.

    The parser decides page numbers and heading chains; later stages never
    reconstruct them. This is the main lesson borrowed from RAG-Anything.
    """

    model_config = ConfigDict(frozen=True)

    doc_id: str
    type: BlockType
    text: str
    table_md: str | None = None  # parser already normalizes tables to Markdown
    caption: str | None = None
    page: int = Field(ge=1)  # 1-based, fixed in the parser
    section_path: list[str]  # heading chain, e.g. ["2 Results", "2.1 Revenue"]
    order: int  # reading order within the document
    origin: Origin = "text"  # text layer, or OCR / vision-LLM transcription


class Chunk(BaseModel):
    """The unit we embed, search, rerank and cite."""

    model_config = ConfigDict(frozen=True)

    id: str  # "doc3:p2:c1", stable across runs
    doc_id: str
    page_start: int = Field(ge=1)  # passed through from Blocks, never reconstructed
    page_end: int = Field(ge=1)
    section_path: list[str]
    type: BlockType
    text: str  # raw: shown to the user and used for quote verification
    embed_text: str  # what the embedder sees (optionally with heading prefix)
    # Default "text" keeps chunks indexed before OCR existed readable.
    origin: Origin = "text"

    @property
    def pages(self) -> list[int]:
        return list(range(self.page_start, self.page_end + 1))


class Hit(BaseModel):
    """A chunk returned by retrieval, with the score of the stage that produced it."""

    chunk: Chunk
    score: float
    # Which stage last scored this hit; useful for debugging and the README GIF.
    source: Literal["dense", "bm25", "hybrid", "rerank"]
