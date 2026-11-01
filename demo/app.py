"""Gradio demo: pick or upload a PDF, ask, get an answer with verified citations.

Run locally:   uv sync --extra cpu --extra demo   (or --extra gpu)
               uv run python demo/app.py
As a Space:    see demo/README.md

Everything below is UI glue. The pipeline is the same one the benchmark measures:
wiring.build_answer_service -> AnswerService.answer (retrieve, rerank, abstain
gate, LLM, Verify). Qdrant runs in-memory; PDFs are indexed the first time they
are asked about.
"""

import os
import re
import tempfile
from io import BytesIO
from pathlib import Path
from typing import Any

import gradio as gr
import pymupdf
from PIL import Image

from docrag.answer.schema import AnswerResult
from docrag.answer.service import AnswerService
from docrag.ingest.indexer import index_documents
from docrag.ingest.ocr import OcrEngine
from docrag.ingest.pages import render_page_png
from docrag.llm.proxy import BudgetExceededError, StructuredOutputError
from docrag.llm.types import ProviderError
from docrag.shared.cache import Cache, bytes_hash
from docrag.shared.config import ExperimentConfig
from docrag.shared.schema import Hit
from docrag.shared.settings import ROOT, Settings
from docrag.wiring import build_answer_service, build_ocr

APP_DIR = Path(__file__).resolve().parent
# Cost and abuse limits (override with environment variables / Space settings).
MAX_QUESTIONS = int(os.environ.get("DOCRAG_DEMO_MAX_QUESTIONS", "10"))  # per session
MAX_PAGES = int(os.environ.get("DOCRAG_DEMO_MAX_PAGES", "60"))  # per uploaded PDF
MAX_UPLOAD_MB = int(os.environ.get("DOCRAG_DEMO_MAX_UPLOAD_MB", "15"))
# The proxy's existing hard cap, per process (= for all visitors together).
os.environ.setdefault("DOCRAG_BUDGET_EUR", "1.0")

_INLINE_ID = re.compile(r"\[([^\]\s]+):p(\d+):c\d+\]")


def sample_pdfs() -> dict[str, Path]:
    """Bundled samples (Space), repo fixtures, and the local benchmark PDFs if any."""
    found: dict[str, Path] = {}
    for folder in (APP_DIR / "samples", ROOT / "fixtures", _settings().pdf_dir):
        if folder.is_dir():
            for pdf in sorted(folder.glob("*.pdf")):
                found.setdefault(pdf.stem, pdf)
    return found


def render_answer(
    result: AnswerResult, hits: list[Hit], threshold: float
) -> tuple[str, list[list[Any]]]:
    """Markdown for the answer + rows for the evidence table. Pure, so it's testable."""
    if result.status == "abstained":
        if result.abstained_before_llm:
            top = f"{hits[0].score:.2f}" if hits else "none"
            why = (
                f"No passage was relevant enough (top rerank score {top} < "
                f"threshold {threshold:.2f}), so no LLM call was made."
            )
        else:
            why = "The model found no answer to exactly this question."
        md = (
            "### Not answerable from this document\n\n"
            f"{why} It abstains instead of guessing."
        )
    else:
        # [doc:p12:c1] -> **[p.12]**: readable for humans, still traceable.
        text = _INLINE_ID.sub(lambda m: f"**[p.{m.group(2)}]**", result.answer)
        label = {
            "supported": "every citation verified",
            "partially_supported": "unverifiable citations were removed",
            "unsupported": "no citation could be verified: treat it as unsourced",
        }[result.status]
        sources = "\n".join(
            f"- **p.{c.page}**: “{c.quote}”" for c in result.verified_citations
        )
        md = f"{text}\n\n*Verify: {label}.*"
        if sources:
            md += f"\n\n**Verified quotes**\n{sources}"
    cited = {c.id for c in result.verified_citations}
    rows = [
        [
            rank,
            h.chunk.page_start,
            "OCR" if h.chunk.origin != "text" else h.chunk.type.upper(),
            round(h.score, 3),
            "yes" if h.chunk.id in cited else "",
            " ".join(h.chunk.text.split())[:160],
        ]
        for rank, h in enumerate(hits, start=1)
    ]
    return md, rows


# --- pipeline (built lazily: importing this module stays cheap) -------------

_state: dict[str, Any] = {}


def _settings() -> Settings:
    if "settings" not in _state:
        # In the repo, use its models/ and .cache/; installed (Space), stay next
        # to the app, because site-packages is not writable there.
        settings = Settings(qdrant_location=":memory:")
        if not (ROOT / "pyproject.toml").exists():
            settings = settings.model_copy(
                update={
                    "models_dir": APP_DIR / ".models",
                    "cache_path": APP_DIR / ".cache" / "docrag.sqlite",
                }
            )
        _state["settings"] = settings
    cached: Settings = _state["settings"]
    return cached


def _pipeline() -> tuple[AnswerService, ExperimentConfig, OcrEngine | None, Cache]:
    if "service" not in _state:
        settings = _settings()
        config = ExperimentConfig.from_yaml(APP_DIR / "config.yaml")
        cache = Cache(settings.cache_path)
        _state["config"] = config
        _state["cache"] = cache
        _state["service"] = build_answer_service(settings, config, cache)
        _state["ocr"] = build_ocr(settings, config, cache)
    return _state["service"], _state["config"], _state["ocr"], _state["cache"]


def _resolve_pdf(choice: str | None, upload: str | None) -> tuple[str, Path]:
    """(doc_id, path). An upload wins over the dropdown."""
    if upload:
        data = Path(upload).read_bytes()
        if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
            raise gr.Error(f"PDF is larger than {MAX_UPLOAD_MB} MB.")
        with pymupdf.open(stream=data, filetype="pdf") as doc:
            if doc.page_count > MAX_PAGES:
                raise gr.Error(
                    f"PDF has {doc.page_count} pages; the demo allows {MAX_PAGES}."
                )
        # Content-addressed: the same file always maps to the same index.
        doc_id = f"upload-{bytes_hash(data)[:12]}"
        path = Path(tempfile.gettempdir()) / "docrag-demo" / f"{doc_id}.pdf"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return doc_id, path
    samples = sample_pdfs()
    if not choice or choice not in samples:
        raise gr.Error("Pick a sample PDF or upload one.")
    return choice, samples[choice]


async def ask(
    choice: str | None, upload: str | None, question: str, asked: int
) -> tuple[str, list[list[Any]], Image.Image | None, int]:
    question = (question or "").strip()
    if not question:
        raise gr.Error("Type a question first.")
    if asked >= MAX_QUESTIONS:
        raise gr.Error(f"This demo allows {MAX_QUESTIONS} questions per session.")
    doc_id, path = _resolve_pdf(choice, upload)
    service, config, ocr, cache = _pipeline()
    store, embeddings = service.retriever.store, service.retriever.embeddings
    if store.count(doc_id) == 0:
        index_documents(
            {doc_id: path},
            store,
            embeddings,
            prefix_headings=config.prefix_headings,
            cache=cache,
            workers=1,
            ocr=ocr,
        )
    try:
        result, hits = await service.answer(question, doc_ids=[doc_id])
    except BudgetExceededError:
        raise gr.Error("The demo's LLM budget is used up for now.") from None
    except (ProviderError, StructuredOutputError) as err:
        raise gr.Error(f"The LLM call failed: {err}") from None
    md, rows = render_answer(result, hits, config.rerank_threshold)
    page = next((c.page for c in result.verified_citations), None)
    image = Image.open(BytesIO(render_page_png(path, page))) if page else None
    return md, rows, image, asked + 1


def build_ui() -> gr.Blocks:
    ui = gr.Blocks(title="docrag-bench demo")
    with ui:
        gr.Markdown(
            "## docrag-bench: answers with verified citations\n"
            "Pick a PDF (or upload one), ask a question. Every citation is checked: "
            "the cited passage must have been retrieved, and the quote must appear "
            "in it verbatim. If the document doesn't contain the answer, it says so."
        )
        asked = gr.State(0)
        with gr.Row():
            with gr.Column(scale=1):
                choice = gr.Dropdown(
                    choices=list(sample_pdfs()), label="Sample PDF", value=None
                )
                upload = gr.File(
                    label=f"...or upload a PDF (max {MAX_UPLOAD_MB} MB, "
                    f"{MAX_PAGES} pages)",
                    file_types=[".pdf"],
                    type="filepath",
                )
                question = gr.Textbox(label="Question", lines=2)
                button = gr.Button("Ask", variant="primary")
            with gr.Column(scale=2):
                answer = gr.Markdown()
                page = gr.Image(label="Page of the first verified citation", height=420)
        evidence = gr.Dataframe(
            headers=["rank", "page", "type", "rerank score", "cited", "text"],
            label="Retrieved evidence (after reranking)",
            wrap=True,
        )
        # One request at a time: the in-memory index is shared by all sessions.
        button.click(
            ask,
            [choice, upload, question, asked],
            [answer, evidence, page, asked],
            concurrency_limit=1,
        )
    return ui


if __name__ == "__main__":
    build_ui().launch(max_file_size=f"{MAX_UPLOAD_MB}mb")
